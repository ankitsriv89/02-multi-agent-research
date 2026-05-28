"""End-to-end test for the LangGraph team with the LLM fully mocked.

We replace `get_langchain_chat_model` so `create_agent` builds a graph
against a fake model that returns canned responses for each node. This
verifies the graph wiring (planner→researcher→writer→critic→finalize)
without making any network calls.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class _ScriptedModel(FakeListChatModel):
    """FakeListChatModel that also supports bind_tools (no-op) so create_agent works."""

    def bind_tools(self, *_args: Any, **_kwargs: Any):
        return self


@pytest.mark.asyncio
async def test_langgraph_team_runs_end_to_end_with_approve():
    # Canned outputs, one per agent turn. The Critic approves on the first
    # pass so the graph follows planner→researcher→writer→critic→finalize.
    responses = [
        "- Subtopic 1\n- Subtopic 2",          # planner
        "- Fact A [https://example.com/a]",     # researcher (no tool calls in mock)
        "Final report body with [1].\n## Sources\n[1] https://example.com/a",  # writer
        "APPROVE: looks good",                  # critic
    ]
    model = _ScriptedModel(responses=responses)

    with patch("app.agents.langgraph_team.get_langchain_chat_model", return_value=model):
        from app.agents.langgraph_team import run_research

        events = [ev async for ev in run_research("What is X?")]

    # Should end with a final + done, no errors.
    types = [ev.type for ev in events]
    assert "error" not in types
    assert types[-1] == "done"
    assert "final" in types

    final = next(ev for ev in events if ev.type == "final")
    assert "Final report body" in (final.content or "")


@pytest.mark.asyncio
async def test_langgraph_team_loops_on_revise_then_approves():
    responses = [
        "- plan",
        "- notes [https://example.com]",
        "draft v1",
        "REVISE: 1. add more detail",  # critic revision #1
        "draft v2",
        "APPROVE: better",             # critic approves
    ]
    model = _ScriptedModel(responses=responses)

    with patch("app.agents.langgraph_team.get_langchain_chat_model", return_value=model):
        from app.agents.langgraph_team import run_research

        events = [ev async for ev in run_research("Topic?")]

    # Writer should appear twice (initial + revision).
    writer_ends = [
        ev for ev in events if ev.type == "agent_end" and ev.agent == "Writer"
    ]
    assert len(writer_ends) == 2
    final = next(ev for ev in events if ev.type == "final")
    assert "draft v2" in (final.content or "")


@pytest.mark.asyncio
async def test_langgraph_team_respects_revision_cap(monkeypatch):
    monkeypatch.setenv("CRITIC_MAX_REVISIONS", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().critic_max_revisions == 1

    # Critic always says REVISE — graph must stop after the cap.
    responses = [
        "- plan",
        "- notes",
        "draft v1",
        "REVISE: 1. fix this",
        "draft v2",
        "REVISE: 2. fix that",  # ignored — we've hit the cap
    ]
    model = _ScriptedModel(responses=responses)

    with patch("app.agents.langgraph_team.get_langchain_chat_model", return_value=model):
        from app.agents.langgraph_team import run_research

        events = [ev async for ev in run_research("Topic?")]

    assert events[-1].type == "done"
    final = next(ev for ev in events if ev.type == "final")
    # Whichever draft is current when the cap hits is accepted as final.
    assert (final.content or "").startswith("draft v")
