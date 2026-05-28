"""LLM factory used by both LangGraph and AutoGen teams.

We expose two helpers:

* `get_langchain_chat_model()` returns a LangChain `BaseChatModel` for the
  LangGraph side (consumed by `langchain.agents.create_agent`).
* `get_autogen_model_client()` returns an `OpenAIChatCompletionClient`
  pointed at Groq's OpenAI-compatible endpoint when LLM_PROVIDER=groq,
  or vanilla OpenAI otherwise.

This keeps both frameworks on the same model so the comparison is fair —
only the orchestration layer differs.
"""
from __future__ import annotations

from app.config import get_settings


def get_langchain_chat_model():
    """Return a LangChain chat model configured per LLM_PROVIDER."""
    settings = get_settings()
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.2,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )


def get_autogen_model_client():
    """Return an AutoGen-compatible OpenAI client.

    Groq exposes an OpenAI-compatible API at https://api.groq.com/openai/v1,
    so we reuse OpenAIChatCompletionClient and just point base_url at Groq.
    autogen-ext requires model_info for non-OpenAI hosts so it knows the
    capability surface (function_calling, vision, etc.).
    """
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    settings = get_settings()
    if settings.llm_provider == "groq":
        return OpenAIChatCompletionClient(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )
    return OpenAIChatCompletionClient(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
