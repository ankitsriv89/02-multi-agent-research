"""Tests for app/report.py — pure functions, fully offline."""
from app.report import (
    build_report_metadata,
    render_report_html,
    render_report_markdown,
    render_report_pdf,
    slugify_query,
)
from app.schemas import Citation


# ── slugify ─────────────────────────────────────────────────────────────────
def test_slugify_basic():
    assert slugify_query("What is the LangGraph 1.x API?") == "what-is-the-langgraph-1-x-api"


def test_slugify_collapses_repeated_punctuation_and_caps_length():
    out = slugify_query("Hello!!!  World??", max_len=11)
    assert out == "hello-world"


def test_slugify_falls_back_when_empty():
    assert slugify_query("???") == "research"
    assert slugify_query("") == "research"


# ── metadata ────────────────────────────────────────────────────────────────
def test_build_report_metadata_shape():
    md_dict = build_report_metadata(
        query="What changed in LangGraph 1.x?",
        framework="langgraph",
        model="openai/gpt-oss-120b",
        elapsed_s=42.7,
        agent_count=4,
        source_count=6,
    )
    expected_keys = {
        "title",
        "framework_label",
        "framework_key",
        "model",
        "timestamp_iso",
        "stats_line",
    }
    assert set(md_dict.keys()) == expected_keys
    assert md_dict["title"] == "What changed in LangGraph 1.x?"
    assert md_dict["framework_label"].startswith("LangGraph")
    # ISO-8601 second-precision: "YYYY-MM-DDTHH:MM:SS"
    assert "T" in md_dict["timestamp_iso"]
    assert md_dict["stats_line"] == "4 agents · 42.7s · 6 sources"


def test_build_report_metadata_unknown_framework_falls_back():
    md_dict = build_report_metadata(
        query="q",
        framework="something-new",
        model="m",
        elapsed_s=1,
        agent_count=1,
        source_count=0,
    )
    assert md_dict["framework_label"] == "something-new"


# ── HTML ────────────────────────────────────────────────────────────────────
def _meta():
    return build_report_metadata(
        query="Test query?",
        framework="langgraph",
        model="openai/gpt-oss-120b",
        elapsed_s=10.0,
        agent_count=4,
        source_count=2,
    )


def test_render_report_html_includes_citations():
    citations = [
        Citation(url="https://example.com/a", title="Example A"),
        Citation(url="https://example.com/b", title="Example B"),
    ]
    html = render_report_html(_meta(), "Some report body.", citations)
    assert "https://example.com/a" in html
    assert "https://example.com/b" in html
    assert "Example A" in html
    assert "Example B" in html
    # Cover header content present
    assert "Test query?" in html
    assert "openai/gpt-oss-120b" in html
    # Self-contained doc
    assert html.startswith("<!doctype html>") or html.startswith("<!DOCTYPE html>")
    assert "<style>" in html


def test_render_report_html_markdown_converted():
    # Bold and a heading should be converted to HTML.
    body = "## A heading\n\nA paragraph with **bold** text and a [link](https://example.com)."
    html = render_report_html(_meta(), body, [])
    assert "<h2>A heading</h2>" in html
    assert "<strong>bold</strong>" in html
    assert 'href="https://example.com"' in html


def test_render_report_html_strips_writer_inline_sources_section():
    # The Writer agent often appends its own "## Sources" section to the body.
    # We must not double-render sources — strip it before HTML conversion.
    body = (
        "Body paragraph one.\n\n"
        "Body paragraph two.\n\n"
        "## Sources\n\n"
        "[1] https://writer-added.example.com"
    )
    citations = [Citation(url="https://structured.example.com", title="From extractor")]
    html = render_report_html(_meta(), body, citations)
    assert "Body paragraph two" in html
    assert "writer-added.example.com" not in html  # stripped
    assert "structured.example.com" in html        # from the structured list


def test_render_report_html_no_citations_omits_footer():
    # The <footer> element only renders when there are citations.
    # (The CSS string contains "footer.sources { ... }", so we look for the
    # actual element tag, not the bare substring.)
    html = render_report_html(_meta(), "Body only.", [])
    assert "<footer" not in html
    assert "<h2>Sources</h2>" not in html


def test_render_report_html_skips_empty_url_citations():
    # Defensive: extractor may give us a Citation with an empty url.
    citations = [
        Citation(url="", title="bad"),
        Citation(url="https://good.example.com", title="good"),
    ]
    html = render_report_html(_meta(), "body", citations)
    assert "https://good.example.com" in html
    assert "<li>bad</li>" not in html


# ── PDF ─────────────────────────────────────────────────────────────────────
def test_render_report_pdf_returns_pdf_bytes():
    html = render_report_html(_meta(), "Just a body.", [])
    pdf = render_report_pdf(html)
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF-")
    # A trivial PDF is at least a few KB; this guards against an empty/broken render.
    assert len(pdf) > 1000


# ── Markdown export ─────────────────────────────────────────────────────────
def test_render_report_markdown_round_trip():
    citations = [
        Citation(url="https://example.com/a", title="A"),
        Citation(url="https://example.com/b", title="B"),
    ]
    out = render_report_markdown(_meta(), "Body text here.", citations)
    # Header content
    assert out.startswith("# Test query?")
    assert "openai/gpt-oss-120b" in out
    assert "LangGraph" in out
    # Body
    assert "Body text here." in out
    # Sources section + URLs
    assert "## Sources" in out
    assert "[A](https://example.com/a)" in out
    assert "[B](https://example.com/b)" in out


def test_render_report_markdown_drops_writer_inline_sources():
    body = "Real body.\n\n## Sources\n[1] https://duplicate.example.com"
    out = render_report_markdown(
        _meta(),
        body,
        [Citation(url="https://kept.example.com", title="K")],
    )
    assert "Real body." in out
    assert "duplicate.example.com" not in out
    assert "kept.example.com" in out
