"""StreamEvent invariants. No LLM access required."""
import pytest
from pydantic import ValidationError

from app.schemas import Citation, StreamEvent


def test_stream_event_round_trip():
    ev = StreamEvent(
        type="final",
        content="Report body",
        citations=[Citation(url="https://example.com", title="Example", snippet="...")],
    )
    dumped = ev.model_dump(exclude_none=True)
    assert dumped["type"] == "final"
    assert dumped["citations"][0]["url"] == "https://example.com"
    # Optional fields shouldn't leak when the producer didn't set them.
    assert "agent" not in dumped


def test_stream_event_rejects_unknown_type():
    with pytest.raises(ValidationError):
        StreamEvent(type="not_a_real_event")


def test_stream_event_accepts_each_documented_type():
    for t in [
        "agent_start",
        "tool_call",
        "tool_result",
        "agent_end",
        "final",
        "error",
        "done",
    ]:
        StreamEvent(type=t)  # should not raise
