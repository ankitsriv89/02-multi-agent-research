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
from urllib.parse import urlparse

import bleach
import markdown as md


# ── Allowlists for HTML sanitization ─────────────────────────────────────────
# The report body comes from an LLM that has just read arbitrary web pages,
# which is a prompt-injection vector: a hostile page can manipulate the LLM
# into emitting <script>, <iframe>, javascript: URIs, etc. We render the
# resulting HTML in two destinations that BOTH execute it:
#   • st.html() in the user's browser (XSS)
#   • WeasyPrint, which can fetch <img src=...> etc. (SSRF, see render_report_pdf)
# Hence the strict allowlist below. Anything not listed is escaped.

_ALLOWED_TAGS: frozenset[str] = frozenset({
    # Block elements the Writer actually uses.
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "blockquote", "pre", "code",
    "table", "thead", "tbody", "tr", "th", "td",
    "hr", "br",
    # Inline formatting.
    "strong", "b", "em", "i", "u", "s", "del", "ins", "sub", "sup",
    "a", "span",
})
_ALLOWED_ATTRS: dict[str, list[str]] = {
    "a": ["href", "title"],
    "th": ["align"],
    "td": ["align"],
    # No "src" anywhere — blocks <img> entirely, which also kills the SSRF
    # surface inside the body. Citation URLs render as <a href>, not images.
}
_ALLOWED_SCHEMES: list[str] = ["http", "https", "mailto"]


def _sanitize_body_html(html: str) -> str:
    return bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_SCHEMES,
        strip=True,
        strip_comments=True,
    )


def _is_safe_url(raw_url: str) -> bool:
    """True only if the URL has an http(s) or mailto scheme."""
    try:
        scheme = urlparse(raw_url).scheme.lower()
    except Exception:
        return False
    return scheme in {"http", "https", "mailto"}


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
        # Block javascript:/data:/vbscript: URIs — these come from regex-mined
        # text in tool messages, an attacker-controlled surface.
        if not _is_safe_url(url):
            continue
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
    raw_body_html = md.markdown(
        body_md,
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html",
    )
    # Sanitize before substituting into the template. See _ALLOWED_TAGS for
    # the rationale — this html is rendered by both st.html() and WeasyPrint.
    body_html = _sanitize_body_html(raw_body_html)
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
def _block_all_url_fetcher(url: str, timeout: int = 10, ssl_context=None) -> dict:
    """WeasyPrint url_fetcher that refuses every outbound fetch.

    Prevents SSRF: WeasyPrint would otherwise fetch <img src=...> etc.,
    which on cloud hosts could hit metadata endpoints (169.254.169.254 etc.)
    or internal services. We embed all our CSS in <style> tags, so there
    are no legitimate fetches to make.
    """
    return {"string": b"", "mime_type": "text/plain"}


def render_report_pdf(html: str) -> bytes:
    """Convert the report HTML into PDF bytes via WeasyPrint.

    Imported lazily because WeasyPrint pulls in heavy native libs at import
    time; tests that don't touch PDF generation should not pay that cost.

    The `url_fetcher` blocks every outbound network fetch (SSRF mitigation).
    """
    from weasyprint import HTML  # noqa: PLC0415  (intentional lazy import)

    return HTML(string=html, url_fetcher=_block_all_url_fetcher).write_pdf()


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
        safe_citations = []
        for c in citations:
            raw_url = getattr(c, "url", None) if not isinstance(c, dict) else c.get("url")
            if not raw_url:
                continue
            url = str(raw_url)
            if not _is_safe_url(url):
                continue
            raw_title = (
                getattr(c, "title", None) if not isinstance(c, dict) else c.get("title")
            )
            title = str(raw_title) if raw_title else url
            safe_citations.append((url, title))
        if safe_citations:
            lines.extend(["", "## Sources", ""])
            for i, (url, title) in enumerate(safe_citations, 1):
                lines.append(f"{i}. [{title}]({url})")
    return "\n".join(lines) + "\n"
