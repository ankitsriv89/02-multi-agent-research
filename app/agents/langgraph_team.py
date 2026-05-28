"""LangGraph multi-agent research team.

Uses LangGraph 1.x `StateGraph` to wire four agents into an explicit
state machine:

    planner → researcher → writer → critic ┐
                              ↑            │
                              └── revise ──┘  (up to N revisions)

Each agent is built with `langchain.agents.create_agent` (the v1
successor to the now-deprecated `langgraph.prebuilt.create_react_agent`).

Streaming uses `graph.astream(..., stream_mode="updates")` which yields a
dict per node-completion. We translate those into the framework-agnostic
`StreamEvent` schema so the FastAPI layer doesn't know which framework
produced the events.
"""
from __future__ import annotations

from typing import Annotated, AsyncIterator, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import get_settings
from app.llm import get_langchain_chat_model
from app.schemas import Citation, StreamEvent
from app.tools.web_fetch import web_fetch
from app.tools.web_search import web_search


# ── Shared state ─────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    """State threaded through every node in the graph.

    `messages` accumulates the conversation across all agents using
    LangGraph's `add_messages` reducer (appends and dedupes by id).
    """

    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    research_notes: str
    draft: str
    critique: str
    revisions: int
    final: str
    citations: list[dict]


# ── Per-agent prompts ────────────────────────────────────────────────────────
PLANNER_PROMPT = (
    "You are the Planner. Given a research query, produce a concise 3-5 bullet "
    "research plan covering what subtopics to investigate and what to verify. "
    "Output only the plan as bullets."
)

RESEARCHER_PROMPT = (
    "You are the Researcher. Execute the plan by calling `web_search` to find "
    "sources, then `web_fetch` on the most promising URLs to read them in full. "
    "Aim for 3-6 sources. After gathering, output a structured notes section "
    "with bullet-pointed facts, each followed by the source URL in brackets, "
    "e.g. '- Fact about X [https://example.com]'. Do not write prose."
)

WRITER_PROMPT = (
    "You are the Writer. Using the research notes, write a clear ~400-word "
    "report answering the original query. Preserve inline citations as "
    "[1], [2], etc., and list the source URLs at the end under '## Sources'. "
    "Be specific and avoid filler."
)

CRITIC_PROMPT = (
    "You are the Critic. Evaluate the draft report against the original query. "
    "Reply with EXACTLY ONE of two formats:\n"
    "  APPROVE: <one-sentence reason>\n"
    "  REVISE: <numbered list of specific changes the writer must make>\n"
    "Approve if the draft is accurate, well-cited, and answers the query. "
    "Only request revisions for substantive issues."
)


# ── Node implementations ─────────────────────────────────────────────────────
async def planner_node(state: ResearchState) -> dict:
    model = get_langchain_chat_model()
    agent = create_agent(model=model, tools=[], system_prompt=PLANNER_PROMPT)
    result = await agent.ainvoke({"messages": [HumanMessage(content=state["query"])]})
    plan = result["messages"][-1].content
    return {
        "plan": plan,
        "messages": [AIMessage(content=f"[Planner]\n{plan}", name="planner")],
    }


async def researcher_node(state: ResearchState) -> dict:
    model = get_langchain_chat_model()
    agent = create_agent(
        model=model,
        tools=[web_search, web_fetch],
        system_prompt=RESEARCHER_PROMPT,
    )
    task = (
        f"Original query: {state['query']}\n\n"
        f"Plan:\n{state['plan']}\n\n"
        "Now gather sources and produce structured notes."
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    notes = result["messages"][-1].content

    # Mine citations from every message in the turn — tool results AND the
    # researcher's own text (some models write URLs inline without calling
    # web_search at all on cached queries).
    import re

    citations: list[dict] = []
    seen: set[str] = set()
    for msg in result["messages"]:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        for url in re.findall(r"https?://[^\s\)\]\"' ]+", content):
            url = url.rstrip(".,;:")
            if url in seen:
                continue
            seen.add(url)
            citations.append({"url": url, "title": url, "snippet": ""})

    return {
        "research_notes": notes,
        "citations": citations[:10],
        "messages": [AIMessage(content=f"[Researcher]\n{notes}", name="researcher")],
    }


async def writer_node(state: ResearchState) -> dict:
    model = get_langchain_chat_model()
    agent = create_agent(model=model, tools=[], system_prompt=WRITER_PROMPT)
    revision_hint = ""
    if state.get("critique") and state["critique"].upper().startswith("REVISE"):
        revision_hint = (
            f"\n\nThe critic requested these revisions to your previous draft:\n"
            f"{state['critique']}\n\nApply them now."
        )
    task = (
        f"Query: {state['query']}\n\n"
        f"Research notes:\n{state['research_notes']}\n"
        f"{revision_hint}"
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    draft = result["messages"][-1].content
    return {
        "draft": draft,
        "messages": [AIMessage(content=f"[Writer]\n{draft}", name="writer")],
    }


async def critic_node(state: ResearchState) -> dict:
    model = get_langchain_chat_model()
    agent = create_agent(model=model, tools=[], system_prompt=CRITIC_PROMPT)
    task = f"Query: {state['query']}\n\nDraft:\n{state['draft']}"
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    critique = result["messages"][-1].content
    return {
        "critique": critique,
        "revisions": state.get("revisions", 0) + 1,
        "messages": [AIMessage(content=f"[Critic]\n{critique}", name="critic")],
    }


def route_after_critic(state: ResearchState) -> str:
    """Loop back to the writer if the critic asked for revisions, else finish.

    Cap the loop at `critic_max_revisions` to bound cost. The writer's
    last draft is accepted as final once we hit the cap.
    """
    settings = get_settings()
    critique = state.get("critique", "")
    if critique.upper().startswith("APPROVE"):
        return "finalize"
    if state.get("revisions", 0) >= settings.critic_max_revisions:
        return "finalize"
    return "writer"


async def finalize_node(state: ResearchState) -> dict:
    return {"final": state["draft"]}


# ── Graph construction ──────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


# ── Streaming adapter for the API layer ─────────────────────────────────────
_NODE_TO_AGENT = {
    "planner": "Planner",
    "researcher": "Researcher",
    "writer": "Writer",
    "critic": "Critic",
}


async def run_research(query: str) -> AsyncIterator[StreamEvent]:
    """Run the graph and emit framework-agnostic StreamEvents."""
    graph = build_graph()
    initial: ResearchState = {
        "query": query,
        "messages": [],
        "plan": "",
        "research_notes": "",
        "draft": "",
        "critique": "",
        "revisions": 0,
        "final": "",
        "citations": [],
    }
    final_state: dict | None = None
    try:
        async for chunk in graph.astream(initial, stream_mode="updates"):
            # Each chunk is {node_name: state_update}
            for node_name, update in chunk.items():
                display = _NODE_TO_AGENT.get(node_name, node_name)
                yield StreamEvent(type="agent_start", agent=display)
                # Pick the most informative field from the update for display.
                content = (
                    update.get("plan")
                    or update.get("research_notes")
                    or update.get("draft")
                    or update.get("critique")
                    or update.get("final")
                    or ""
                )
                if content:
                    yield StreamEvent(
                        type="agent_end", agent=display, content=content
                    )
                final_state = update
        # Emit final report with citations.
        if final_state and final_state.get("final"):
            citations_raw = final_state.get("citations") or []
            citations = [Citation(**c) for c in citations_raw]
            yield StreamEvent(
                type="final",
                content=final_state["final"],
                citations=citations,
            )
        yield StreamEvent(type="done")
    except Exception as e:
        yield StreamEvent(type="error", content=str(e))
