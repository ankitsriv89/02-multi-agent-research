"""Tavily web search — wrapped as a plain async function.

We expose a `web_search(query)` callable that works for both frameworks:
LangGraph's `create_agent` accepts plain callables and converts them to
tools via type hints + docstring; AutoGen's AssistantAgent does the same.
This avoids carrying around two different tool wrappers for the same op.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from app.config import get_settings


async def web_search(query: str) -> list[dict[str, Any]]:
    """Search the web for up-to-date information.

    Args:
        query: A natural-language search query, e.g. "GPT-5 release date".

    Returns:
        Up to N results, each a dict with `url`, `title`, and `content`
        (the snippet). Returns [] on any error so the agent can recover.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY missing — web_search returning empty results")
        return []

    # Import lazily so tests that don't touch this tool don't need tavily installed.
    from langchain_tavily import TavilySearch

    tool = TavilySearch(
        max_results=settings.web_search_max_results,
        tavily_api_key=settings.tavily_api_key,
        topic="general",
    )
    try:
        result = await asyncio.to_thread(tool.invoke, {"query": query})
    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return []

    # TavilySearch returns either a dict with "results" or a list directly.
    if isinstance(result, dict):
        items = result.get("results", [])
    else:
        items = result if isinstance(result, list) else []
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", "")[:500],
        }
        for r in items
    ]
