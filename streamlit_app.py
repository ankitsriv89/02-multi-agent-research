"""Streamlit UI for the multi-agent research system.

Single-process app: imports the LangGraph and AutoGen teams directly and
iterates the same StreamEvent generator they emit. No FastAPI, no SSE
proxy — Streamlit handles the streaming via its native chat primitives.

Layout:

    ┌─────────────────────────────────────────────────────────┐
    │ Sidebar: framework toggle, model info, examples         │
    ├─────────────────────────────────────────────────────────┤
    │ Main:                                                   │
    │   Query input                                           │
    │   Two columns:                                          │
    │     Left — live agent trace (one expander per turn)     │
    │     Right — final report + citations                    │
    └─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal

import streamlit as st

if TYPE_CHECKING:
    from app.schemas import StreamEvent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# HF Spaces stores secrets at /etc/secrets and as env vars; both paths work
# with pydantic-settings. We don't need to do anything special here.

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Multi-Agent Research")
    st.caption(
        "Same task, two frameworks. Watch four agents (Planner, Researcher, "
        "Writer, Critic) collaborate to produce a cited report."
    )

    framework: Literal["langgraph", "autogen"] = st.radio(
        "Agent framework",
        options=["langgraph", "autogen"],
        format_func=lambda x: "LangGraph (explicit state machine)"
        if x == "langgraph"
        else "AutoGen (LLM-routed group chat)",
        help="Both teams have identical roles and tools. Only the orchestration differs.",
    )

    with st.expander("How it works"):
        st.markdown(
            """
**LangGraph** — explicit `StateGraph`: planner → researcher → writer → critic,
with a conditional edge that loops back to writer on `REVISE` (capped).

**AutoGen** — `SelectorGroupChat`: an LLM picks the next speaker from the
agent transcript each turn. Termination fires when someone says `APPROVE`
or the message cap is hit.

See `docs/COMPARISON.md` in the repo for the head-to-head.
            """
        )

    with st.expander("Configuration"):
        # Show what the backend actually loaded, after env precedence resolves.
        from app.config import get_settings

        s = get_settings()
        st.write(f"**Provider:** `{s.llm_provider}`")
        st.write(f"**Model:** `{s.groq_model if s.llm_provider == 'groq' else s.openai_model}`")
        st.write(f"**Revision cap:** {s.critic_max_revisions}")
        st.write(f"**Search results:** {s.web_search_max_results}")
        missing = []
        if s.llm_provider == "groq" and not s.groq_api_key:
            missing.append("GROQ_API_KEY")
        if s.llm_provider == "openai" and not s.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not s.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            st.error(f"Missing secrets: {', '.join(missing)}")

    st.markdown("---")
    st.caption(
        "Built by Ankit · "
        "[GitHub](https://github.com/anksr2018) · "
        "Project 2/8"
    )

# ── Main pane ────────────────────────────────────────────────────────────────
st.markdown("# 🔎 Multi-Agent Research")
st.caption(
    f"Running on **{framework.upper()}** — change in the sidebar to compare."
)

example_queries = [
    "What is the LangGraph 1.x StateGraph API?",
    "What are the most cited LLM evaluation benchmarks in 2026?",
    "How does QLoRA differ from LoRA, and when is each preferred?",
]

# Stateful query input so example-click reruns work cleanly.
if "query" not in st.session_state:
    st.session_state.query = ""

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Research query",
        value=st.session_state.query,
        placeholder=example_queries[0],
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("Research", type="primary", use_container_width=True)

# Example chips.
with st.expander("Try an example"):
    for ex in example_queries:
        if st.button(ex, key=f"ex-{ex[:30]}"):
            st.session_state.query = ex
            st.rerun()

if not run or not query.strip():
    st.info(
        "Enter a question and hit **Research**. A four-agent team will "
        "plan, search the web, write a cited report, and iterate with a critic."
    )
    st.stop()


# ── Run the team and stream events into the UI ───────────────────────────────
def _team_runner(framework: str, query: str):
    """Return the async generator that yields StreamEvents."""
    if framework == "langgraph":
        from app.agents.langgraph_team import run_research
    else:
        from app.agents.autogen_team import run_research
    return run_research(query)


async def _collect_events(framework: str, query: str, ui_update):
    async for ev in _team_runner(framework, query):
        ui_update(ev)


# Layout: left = trace, right = report.
trace_col, report_col = st.columns([1, 1], gap="large")
with trace_col:
    st.markdown("### Agent trace")
    trace_container = st.container()
with report_col:
    st.markdown("### Report")
    report_container = st.container()
    citations_container = st.container()

# State we accumulate as events stream in.
events: list = []
agent_blocks: dict[str, Any] = {}  # key: "<agent>#<turn-idx>" → st.expander
current_turn: dict[str, int] = {}     # agent -> how many turns it's taken
final_text = ""
final_citations: list = []
error_text: str | None = None
start = time.perf_counter()


_AGENT_EMOJI = {
    "Planner": "🗺️", "planner": "🗺️",
    "Researcher": "🔍", "researcher": "🔍",
    "Writer": "✍️", "writer": "✍️",
    "Critic": "🧐", "critic": "🧐",
    "finalize": "✅",
}


def _render_event(ev: "StreamEvent") -> None:
    """Push one event into the UI."""
    global final_text, final_citations, error_text  # noqa: PLW0603
    events.append(ev)

    if ev.type == "agent_start":
        # Create a new expander for this turn. Some frameworks emit start
        # without content; the body fills in on agent_end / tool_call.
        agent = ev.agent or "agent"
        current_turn[agent] = current_turn.get(agent, 0) + 1
        key = f"{agent}#{current_turn[agent]}"
        with trace_container:
            exp = st.expander(
                f"{_AGENT_EMOJI.get(agent, '🤖')} {agent} — turn {current_turn[agent]}",
                expanded=True,
            )
            agent_blocks[key] = exp

    elif ev.type == "tool_call":
        agent = ev.agent or "agent"
        key = f"{agent}#{current_turn.get(agent, 1)}"
        block = agent_blocks.get(key)
        if block is not None:
            with block:
                st.markdown(f"**🔧 tool_call** `{ev.tool}`")
                if ev.args:
                    st.json(ev.args)

    elif ev.type == "tool_result":
        agent = ev.agent or "agent"
        key = f"{agent}#{current_turn.get(agent, 1)}"
        block = agent_blocks.get(key)
        if block is not None:
            with block:
                st.markdown(f"**📄 tool_result** `{ev.tool}`")
                st.code((ev.content or "")[:500], language="text")

    elif ev.type == "agent_end":
        agent = ev.agent or "agent"
        key = f"{agent}#{current_turn.get(agent, 1)}"
        block = agent_blocks.get(key)
        if block is not None:
            with block:
                st.markdown(ev.content or "_(no content)_")

    elif ev.type == "final":
        final_text = ev.content or ""
        final_citations = ev.citations or []
        with report_container:
            st.markdown(final_text)
        with citations_container:
            if final_citations:
                st.markdown("---")
                st.markdown("### Sources")
                for i, c in enumerate(final_citations, 1):
                    title = c.title if hasattr(c, "title") else c.get("title", c.get("url"))
                    url = c.url if hasattr(c, "url") else c.get("url")
                    st.markdown(f"[{i}] [{title}]({url})")

    elif ev.type == "error":
        error_text = ev.content or "Unknown error"
        st.error(f"Agent error: {error_text}")


# Run the async generator inside Streamlit's sync flow.
status = st.status(f"Running {framework.upper()} team…", expanded=False)
try:
    asyncio.run(_collect_events(framework, query, _render_event))
    elapsed = time.perf_counter() - start
    if error_text:
        status.update(label=f"Failed in {elapsed:.1f}s", state="error")
    elif final_text:
        status.update(
            label=f"Done in {elapsed:.1f}s · {len(events)} events · "
            f"{len(final_citations)} sources",
            state="complete",
        )
    else:
        status.update(label=f"Ended in {elapsed:.1f}s (no report)", state="error")
except Exception as e:
    status.update(label="Crashed", state="error")
    st.exception(e)
