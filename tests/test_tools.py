"""Tool-level tests — exercise web_search and web_fetch with mocked I/O."""
import httpx
import pytest
import respx

from app.tools.web_fetch import web_fetch
from app.tools.web_search import web_search


# ── web_search ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_web_search_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()
    results = await web_search("anything")
    assert results == []


@pytest.mark.asyncio
async def test_web_search_normalises_tavily_response():
    # langchain_tavily may not be installed in the dev env; inject a stub
    # module before web_search imports it.
    import sys
    import types

    fake = {
        "results": [
            {
                "url": "https://example.com/a",
                "title": "Doc A",
                "content": "Body A" + "x" * 1000,
            },
            {"url": "https://example.com/b", "title": "Doc B", "content": "Body B"},
        ]
    }

    class _FakeTavilySearch:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, _input):
            return fake

    stub = types.ModuleType("langchain_tavily")
    stub.TavilySearch = _FakeTavilySearch  # type: ignore[attr-defined]
    sys.modules["langchain_tavily"] = stub
    try:
        results = await web_search("query")
    finally:
        sys.modules.pop("langchain_tavily", None)

    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/a"
    # Snippet should be truncated to 500 chars.
    assert len(results[0]["content"]) == 500


# ── web_fetch ───────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_extracts_article_text():
    # Readability needs a substantive article body to win over chrome,
    # so we pad the article with real-looking sentences.
    body = " ".join(
        [
            "This is a substantial article body that readability will identify.",
            "It contains multiple sentences with meaningful content.",
            "The point is to give readability enough signal to pick it.",
        ]
        * 6
    )
    html = f"""
    <html><body>
        <nav>Site nav menu items go here links and stuff</nav>
        <article>
            <h1>Title of the Article</h1>
            <p>Article paragraph one. {body}</p>
            <p>Article paragraph two. {body}</p>
        </article>
        <footer>Site footer text copyright info</footer>
    </body></html>
    """
    respx.get("https://example.com/article").mock(
        return_value=httpx.Response(200, text=html)
    )
    text = await web_fetch("https://example.com/article")
    # Article body must be present; the readability heuristic isn't perfect
    # on synthetic HTML so we don't assert that boilerplate is fully gone.
    assert "Article paragraph one" in text
    assert "Article paragraph two" in text
    assert "Title of the Article" in text


@pytest.mark.asyncio
async def test_web_fetch_rejects_non_http_urls():
    assert await web_fetch("file:///etc/passwd") == ""
    assert await web_fetch("javascript:alert(1)") == ""


@pytest.mark.asyncio
@respx.mock
async def test_web_fetch_swallows_network_errors():
    respx.get("https://broken.example").mock(side_effect=httpx.ConnectError("boom"))
    assert await web_fetch("https://broken.example") == ""
