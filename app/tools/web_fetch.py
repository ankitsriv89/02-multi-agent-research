"""Fetch a URL and return readable article text.

Used by the Researcher agent to read a page after the Tavily snippet
proves promising. Uses readability-lxml to strip boilerplate.
"""
from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from readability import Document

from app.config import get_settings

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MultiAgentResearchBot/1.0; "
        "+https://github.com/ankit/jobs-prjcts)"
    )
}


async def web_fetch(url: str) -> str:
    """Fetch a URL and return its main article text, cleaned of boilerplate.

    Args:
        url: An absolute http(s) URL.

    Returns:
        Cleaned text content, truncated to a sane budget. Empty string on error.
    """
    settings = get_settings()
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        async with httpx.AsyncClient(
            timeout=settings.web_fetch_timeout_s,
            follow_redirects=True,
            headers=_HEADERS,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.warning(f"web_fetch({url}) failed: {e}")
        return ""

    try:
        doc = Document(html)
        cleaned_html = doc.summary()
        text = BeautifulSoup(cleaned_html, "lxml").get_text(separator="\n", strip=True)
    except Exception as e:
        logger.warning(f"web_fetch parse failed for {url}: {e}")
        text = BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)

    return text[: settings.web_fetch_max_chars]
