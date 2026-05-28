"""Shared pytest fixtures.

We set fake API keys at import time so `Settings()` succeeds even on a
machine without secrets — every test that hits an LLM mocks the call.
"""
import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("LLM_PROVIDER", "groq")

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test gets a fresh Settings() so env overrides take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
