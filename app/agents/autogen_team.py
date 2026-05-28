"""AutoGen multi-agent research team.

Mirror of the LangGraph team — same four roles, same Groq model — but
using AutoGen's `SelectorGroupChat` for conversation-driven routing.

`SelectorGroupChat` uses an LLM to pick the next speaker based on the
running transcript and each agent's description. That contrasts sharply
with LangGraph's explicit conditional edges, which is the headline of
the LangGraph-vs-AutoGen comparison in docs/COMPARISON.md.

Termination: stop when the Critic says 'APPROVE' OR we hit max messages.
"""
from __future__ import annotations

from typing import AsyncIterator

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from autogen_agentchat.teams import SelectorGroupChat

from app.config import get_settings
from app.llm import get_autogen_model_client
from app.schemas import Citation, StreamEvent
from app.tools.web_fetch import web_fetch
from app.tools.web_search import web_search


PLANNER_SYS = (
    "You are the Planner. Produce a 3-5 bullet research plan for the user's "
    "query. After your plan, do not speak again."
)
RESEARCHER_SYS = (
    "You are the Researcher. Follow the Planner's plan: call `web_search`, "
    "then `web_fetch` on promising URLs. After 3-6 sources, output bullet "
    "notes with each fact followed by [URL]. Do not write prose."
)
WRITER_SYS = (
    "You are the Writer. Using the Researcher's notes, write a ~400-word "
    "report with inline [1], [2] citations and a '## Sources' list at the "
    "end. If the Critic requests revisions, address them and re-post the "
    "full revised report."
)
CRITIC_SYS = (
    "You are the Critic. Evaluate the latest report. Reply with EITHER "
    "'APPROVE: <reason>' (this ends the conversation) OR "
    "'REVISE: <numbered changes>'. Be strict but only require revisions "
    "for substantive issues."
)


def build_team() -> SelectorGroupChat:
    settings = get_settings()
    model_client = get_autogen_model_client()

    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message=PLANNER_SYS,
        description="Produces the research plan. Speaks first, then yields.",
    )
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        tools=[web_search, web_fetch],
        system_message=RESEARCHER_SYS,
        description="Calls web tools to gather sources. Speaks after the planner.",
        reflect_on_tool_use=True,
    )
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message=WRITER_SYS,
        description="Writes/revises the report from the researcher's notes.",
    )
    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message=CRITIC_SYS,
        description="Reviews the writer's draft. Approves or requests revisions.",
    )

    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(
        # Generous ceiling: planner + researcher (+tool turns) + writer + critic
        # + (revision_cap * (writer + critic)).
        max_messages=10 + 2 * settings.critic_max_revisions
    )
    return SelectorGroupChat(
        participants=[planner, researcher, writer, critic],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )


async def run_research(query: str) -> AsyncIterator[StreamEvent]:
    """Run the AutoGen team and emit framework-agnostic StreamEvents."""
    team = build_team()
    last_writer_text = ""
    citations: list[Citation] = []
    seen_urls: set[str] = set()

    try:
        async for event in team.run_stream(task=query):
            # Terminal marker from AutoGen — TaskResult arrives at the end.
            if isinstance(event, TaskResult):
                break

            agent = getattr(event, "source", None)

            if isinstance(event, TextMessage):
                content = event.content if isinstance(event.content, str) else str(event.content)
                yield StreamEvent(type="agent_start", agent=agent)
                yield StreamEvent(type="agent_end", agent=agent, content=content)
                if agent == "writer":
                    last_writer_text = content

            elif isinstance(event, ToolCallRequestEvent):
                for call in event.content:
                    yield StreamEvent(
                        type="tool_call",
                        agent=agent,
                        tool=getattr(call, "name", None),
                        args=getattr(call, "arguments", None)
                        if isinstance(getattr(call, "arguments", None), dict)
                        else None,
                    )

            elif isinstance(event, ToolCallExecutionEvent):
                for result in event.content:
                    text = str(getattr(result, "content", ""))
                    # Mine URLs for the citations panel.
                    import re

                    for url in re.findall(r"https?://[^\s\)\]\"']+", text):
                        if url not in seen_urls:
                            seen_urls.add(url)
                            citations.append(
                                Citation(url=url, title=url, snippet="")
                            )
                    yield StreamEvent(
                        type="tool_result",
                        agent=agent,
                        tool=getattr(result, "name", None),
                        content=text[:500],
                    )

        if last_writer_text:
            yield StreamEvent(
                type="final",
                content=last_writer_text,
                citations=citations[:10],
            )
        yield StreamEvent(type="done")
    except Exception as e:
        yield StreamEvent(type="error", content=str(e))
