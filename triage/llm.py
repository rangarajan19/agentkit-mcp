"""Thin Gemini wrapper. Keep all provider-specific code here so it can be swapped."""
import time

from google import genai
from google.genai import types

from . import config

_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _retry(fn, attempts: int = 5):
    """Free tier returns 429 when rate limited; back off and retry."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - SDK raises several error types
            if i == attempts - 1 or "429" not in str(e):
                raise
            time.sleep(2 ** i)


def embed(texts: list[str]) -> list[list[float]]:
    res = _retry(lambda: client().models.embed_content(
        model=config.GEMINI_EMBED_MODEL, contents=texts))
    return [e.values for e in res.embeddings]


def run_agent(system: str, prompt: str, tools: list) -> str:
    """Run one agent turn. The SDK executes the python callables in `tools`
    automatically (function-calling loop) and returns the final text."""
    res = _retry(lambda: client().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            tools=tools,
            temperature=0.2,
        ),
    ))
    return res.text or ""
