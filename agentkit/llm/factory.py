"""Build an LLM from a short spec string.

  "gemini:gemini-3.5-flash,gemini-3.1-flash-lite"   Gemini (comma = fallback chain)
  "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"  OpenRouter model
  "gemini-3.5-flash" or None                         Gemini (default)
"""
from __future__ import annotations

from .base import LLM
from .gemini import GeminiLLM
from .openrouter import OpenRouterLLM


def make_llm(spec: str | None = None) -> LLM:
    if not spec:
        return GeminiLLM()
    provider, sep, rest = spec.partition(":")
    if sep and provider == "openrouter":
        return OpenRouterLLM(model=rest)
    if sep and provider == "gemini":
        return GeminiLLM(model=rest)
    return GeminiLLM(model=spec)
