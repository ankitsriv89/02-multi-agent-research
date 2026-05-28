"""Centralised settings loaded from environment / .env.

Reuses the Project 1 pattern: pydantic-settings with a single Settings()
instance imported wherever config is needed.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM routing ──────────────────────────────────────────────────────────
    llm_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: str = ""
    openai_api_key: str = ""

    # gpt-oss-120b is OpenAI's open-weights model on Groq, trained for the
    # OpenAI tool-call format. llama-3.3-70b emits malformed <function=name {...}>
    # blobs on multi-tool prompts and Groq's server rejects them — confirmed
    # in production. Don't change this unless you've tested tool-calling.
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_models: list[str] = Field(
        default_factory=lambda: [
            "openai/gpt-oss-20b",
            "llama-3.1-8b-instant",
        ]
    )
    openai_model: str = "gpt-4o-mini"

    # ── Tools ────────────────────────────────────────────────────────────────
    tavily_api_key: str = ""
    web_search_max_results: int = 5
    web_fetch_timeout_s: float = 15.0
    web_fetch_max_chars: int = 8000

    # ── Agent loop bounds (cost & latency guardrails) ────────────────────────
    max_agent_iterations: int = 8
    critic_max_revisions: int = 2

    # ── App ──────────────────────────────────────────────────────────────────
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
