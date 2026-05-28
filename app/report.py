"""Report rendering: HTML, PDF, and Markdown export.

Pure functions only — no Streamlit imports — so the module is fully
unit-testable and reusable from any caller. The Streamlit layer calls
into these functions to build the on-screen pretty render AND the
download_button payloads.

The HTML produced by `render_report_html` doubles as:
  • the source of truth for the PDF (passed through WeasyPrint)
  • the on-screen render (passed to `st.html()` in the UI)

Same look, both surfaces.
"""
from __future__ import annotations

import re
from datetime import datetime

import markdown as md


# ── Slug ─────────────────────────────────────────────────────────────────────
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify_query(query: str, max_len: int = 50) -> str:
    """Turn a query into a filesystem-safe slug.

    Examples:
        "What is the LangGraph 1.x API?" → "what-is-the-langgraph-1-x-api"
        "  Empty?? "                     → "empty"
    """
    s = _SLUG_STRIP.sub("-", query.lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "research"


# ── Metadata header ──────────────────────────────────────────────────────────
_FRAMEWORK_LABELS = {
    "langgraph": "LangGraph (explicit state machine)",
    "autogen": "AutoGen (LLM-routed group chat)",
}


def build_report_metadata(
    query: str,
    framework: str,
    model: str,
    elapsed_s: float,
    agent_count: int,
    source_count: int,
) -> dict[str, str]:
    """Assemble the header block shown above the report body.

    Returns a flat dict so it's trivially serializable and easy to test.
    """
    return {
        "title": query.strip(),
        "framework_label": _FRAMEWORK_LABELS.get(framework, framework),
        "framework_key": framework,
        "model": model,
        "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        "stats_line": (
            f"{agent_count} agents · {elapsed_s:.1f}s · {source_count} sources"
        ),
    }


# ── HTML (used for on-screen + PDF) ──────────────────────────────────────────
# Embedded so the output is a single self-contained document.
# Charter / Inter are widely available; Liberation Serif/Sans are the
# Debian fallbacks we ship via fonts-liberation in the Dockerfile.
_CSS = """
@page {
  size: A4;
  margin: 2cm 2cm 2.5cm 2cm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: Inter, "Liberation Sans", system-ui, sans-serif;
    font-size: 9pt;
    color: #888;
  }
}
:root {
  --accent: #2563eb;
  --ink: #1a1a1a;
  --muted: #6b7280;
  --rule: #e5e7eb;
}
html, body {
  font-family: Charter, Georgia, "Liberation Serif", "Times New Roman", serif;
  font-size: 11pt;
  line-height: 1.55;
  color: var(--ink);
  margin: 0;
  padding: 0;
}
.report-wrap {
  max-width: 760px;
  margin: 0 auto;
  padding: 0 0 2rem 0;
}
header.cover {
  border-bottom: 2px solid var(--accent);
  padding-bottom: 0.75rem;
  margin-bottom: 1.25rem;
}
.eyebrow {
  font-family: Inter, "Liberation Sans", system-ui, sans-serif;
  font-size: 9pt;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 0.35rem;
}
h1.title {
  font-family: Inter, "Liberation Sans", system-ui, sans-serif;
  font-size: 20pt;
  font-weight: 700;
  line-height: 1.2;
  margin: 0 0 0.6rem 0;
  color: var(--ink);
}
.meta {
  font-family: Inter, "Liberation Sans", system-ui, sans-serif;
  font-size: 9.5pt;
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 0 1.25rem;
}
.meta span strong { color: var(--ink); font-weight: 600; }
main.body h2 {
  font-family: Inter, "Liberation Sans", system-ui, sans-serif;
  font-size: 13.5pt;
  font-weight: 700;
  margin: 1.6rem 0 0.5rem 0;
  color: var(--ink);
}
main.body h3 {
  font-family: Inter, "Liberation Sans", system-ui, sans-serif;
  font-size: 11.5pt;
  font-weight: 600;
  margin: 1.2rem 0 0.3rem 0;
}
main.body p { margin: 0 0 0.75rem 0; }
main.body ul, main.body ol { margin: 0.25rem 0 0.85rem 1.5rem; padding: 0; }
main.body li { margin-bottom: 0.25rem; }
main.body code {
  font-family: "SF Mono", Consolas, "Liberation Mono", monospace;
  font-size: 0.9em;
  background: #f3f4f6;
  padding: 0.05em 0.3em;
  border-radius: 3px;
}
main.body pre {
  background: #f9fafb;
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 0.6rem 0.8rem;
  overflow-x: auto;
  font-size: 9.5pt;
}
main.body a { color: var(--accent); text-decoration: none; }
main.body a:hover { text-decoration: underline; }
footer.sources {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--rule);
  font-family: Inter, "Liberation Sans", system-ui, sans-serif;
  font-size: 9.5pt;
}
footer.sources h2 {
  font-size: 11pt;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  color: var(--ink);
  letter-spacing: 0.02em;
}
footer.sources ol {
  margin: 0;
  padding-left: 1.5rem;
  color: var(--muted);
}
footer.sources li { margin-bottom: 0.35rem; word-break: break-all; }
footer.sources a { color: var(--accent); text-decoration: none; }
"""

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_escaped}</title>
<style>{css}</style>
</head>
<body>
<div class="report-wrap">
  <header class="cover">
    <div class="eyebrow">Multi-agent research report</div>
    <h1 class="title">{title_escaped}</h1>
    <div class="meta">
      <span><strong>Framework:</strong> {framework_label_escaped}</span>
      <span><strong>Model:</strong> {model_escaped}</span>
      <span><strong>Generated:</strong> {timestamp}</span>
      <span>{stats_line}</span>
    </div>
  </header>
  <main class="body">{body_html}</main>
  {sources_block}
</div>
</body>
</html>"""


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _sources_html(citations: list) -> str:
    """Build the <footer> sources block, or empty string if no citations."""
    if not citations:
        return ""
    items = []
    for c in citations:
        # Tolerate both Citation pydantic objects and plain dicts.
        raw_url = getattr(c, "url", None) if not isinstance(c, dict) else c.get("url")
        if not raw_url:
            continue
        url = str(raw_url)
        raw_title = (
            getattr(c, "title", None) if not isinstance(c, dict) else c.get("title")
        )
        title = str(raw_title) if raw_title else url
        items.append(
            f'<li><a href="{_escape(url)}">{_escape(title)}</a></li>'
        )
    if not items:
        return ""
    return (
        '<footer class="sources">'
        "<h2>Sources</h2>"
        "<ol>" + "".join(items) + "</ol>"
        "</footer>"
    )


# The body may already contain a "## Sources" section emitted by the
# Writer agent — we strip it so we don't render two sources blocks.
_SOURCES_HEADER_RE = re.compile(
    r"\n##\s*Sources\s*\n.*",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_inline_sources(body_md: str) -> str:
    return _SOURCES_HEADER_RE.sub("", body_md).rstrip()


def render_report_html(
    metadata: dict,
    body_markdown: str,
    citations: list,
) -> str:
    """Build a complete, self-contained HTML document.

    Used as the source for both the PDF (via WeasyPrint) and the on-screen
    render (via `st.html()` in the Streamlit layer).
    """
    body_md = _strip_inline_sources(body_markdown or "")
    body_html = md.markdown(
        body_md,
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html",
    )
    return _HTML_TEMPLATE.format(
        title_escaped=_escape(metadata["title"]),
        framework_label_escaped=_escape(metadata["framework_label"]),
        model_escaped=_escape(metadata["model"]),
        timestamp=_escape(metadata["timestamp_iso"].replace("T", " ")),
        stats_line=_escape(metadata["stats_line"]),
        body_html=body_html,
        sources_block=_sources_html(citations),
        css=_CSS,
    )


# ── PDF ──────────────────────────────────────────────────────────────────────
def render_report_pdf(html: str) -> bytes:
    """Convert the report HTML into PDF bytes via WeasyPrint.

    Imported lazily because WeasyPrint pulls in heavy native libs at import
    time; tests that don't touch PDF generation should not pay that cost.
    """
    from weasyprint import HTML  # noqa: PLC0415  (intentional lazy import)

    return HTML(string=html).write_pdf()


# ── Markdown export ──────────────────────────────────────────────────────────
def render_report_markdown(
    metadata: dict,
    body_markdown: str,
    citations: list,
) -> str:
    """Plain-text markdown export. Mirrors the HTML structure."""
    body_md = _strip_inline_sources(body_markdown or "")
    lines = [
        f"# {metadata['title']}",
        "",
        f"- **Framework:** {metadata['framework_label']}",
        f"- **Model:** {metadata['model']}",
        f"- **Generated:** {metadata['timestamp_iso'].replace('T', ' ')}",
        f"- {metadata['stats_line']}",
        "",
        "---",
        "",
        body_md,
    ]
    if citations:
        lines.extend(["", "## Sources", ""])
        for i, c in enumerate(citations, 1):
            raw_url = getattr(c, "url", None) if not isinstance(c, dict) else c.get("url")
            if not raw_url:
                continue
            url = str(raw_url)
            raw_title = (
                getattr(c, "title", None) if not isinstance(c, dict) else c.get("title")
            )
            title = str(raw_title) if raw_title else url
            lines.append(f"{i}. [{title}]({url})")
    return "\n".join(lines) + "\n"
