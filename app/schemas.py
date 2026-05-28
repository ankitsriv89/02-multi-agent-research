"""Internal event model used by both teams.

A single async generator carries the full lifecycle of a research run:

    agent_start  → which agent began a turn
    tool_call    → tool name + arguments
    tool_result  → tool output (often truncated for display)
    agent_end    → agent finished; content has the final text for that turn
    final        → the writer's accepted report (with citations)
    error        → terminal failure; stream ends after this
    done         → terminal success marker

The Streamlit UI consumes these events and renders them into the agent
trace pane and the report pane.
"""
from typing import Any, Literal

from pydantic import BaseModel


Framework = Literal["langgraph", "autogen"]


class Citation(BaseModel):
    url: str
    title: str
    snippet: str = ""


class StreamEvent(BaseModel):
    type: Literal[
        "agent_start",
        "tool_call",
        "tool_result",
        "agent_end",
        "final",
        "error",
        "done",
    ]
    agent: str | None = None
    content: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None
    citations: list[Citation] | None = None
