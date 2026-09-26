"""OpenRouter adapter (OpenAI-compatible chat API). Gives access to many models, including free ones.

Set OPENROUTER_API_KEY. Free models end in ':free', e.g. nvidia/nemotron-3-ultra-550b-a55b:free.
"""
from __future__ import annotations

import json
import os
import sys
import time

import openai
from dotenv import load_dotenv

from ..types import LLMResponse, Message, ToolCall, ToolSpec

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = (408, 429, 500, 502, 503, 504)


class ProviderError(RuntimeError):
    """OpenRouter sometimes returns HTTP 200 with an error payload; carry its code so we can retry."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def is_temporary(e: Exception) -> bool:
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return getattr(e, "status_code", None) in RETRY_STATUS


def _retry(fn, attempts: int = 4):
    """Free models are often rate limited or briefly unavailable: back off and retry."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_temporary(e) or i == attempts - 1:
                raise
            print(f"[agentkit] temporary error ({getattr(e, 'status_code', type(e).__name__)}), "
                  f"retrying in {3 * 2 ** i}s", file=sys.stderr)
            time.sleep(3 * 2 ** i)


class OpenRouterLLM:
    def __init__(self, model: str, api_key: str | None = None, timeout: float = 180.0):
        self.model = model
        self.client = openai.OpenAI(
            base_url=BASE_URL, api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            timeout=timeout, max_retries=0)  # we do our own retrying

    # --- conversion ---------------------------------------------------------
    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict]:
        return [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools]

    @staticmethod
    def _messages(system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        pending_ids: list[str] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                msg = m.raw if isinstance(m.raw, dict) else {"role": "assistant", "content": m.text or None}
                out.append(msg)
                pending_ids = [c["id"] for c in msg.get("tool_calls") or []]
            elif m.role == "tool":
                # results answer the previous assistant turn's tool calls, in the same order
                for call_id, (_name, result) in zip(pending_ids, m.results):
                    out.append({"role": "tool", "tool_call_id": call_id, "content": result})
        return out

    def _create(self, kwargs: dict):
        res = self.client.chat.completions.create(**kwargs)
        if not res.choices:
            err = getattr(res, "error", None) or {}
            code = err.get("code") if isinstance(err, dict) else getattr(err, "code", None)
            raise ProviderError(f"empty response from {self.model}: {err or res}",
                                code if isinstance(code, int) else None)
        return res

    # --- LLM protocol -------------------------------------------------------
    def generate(self, system: str, messages: list[Message],
                 tools: list[ToolSpec]) -> LLMResponse:
        kwargs = {"model": self.model, "messages": self._messages(system, messages),
                  "temperature": 0.2}
        if tools:
            kwargs["tools"] = self._tools(tools)
        res = _retry(lambda: self._create(kwargs))
        msg = res.choices[0].message

        calls, raw_calls = [], []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": c.function.arguments}  # the tool call will fail; model can retry
            calls.append(ToolCall(name=c.function.name, args=args))
            raw_calls.append({"id": c.id, "type": "function",
                              "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}})
        raw = {"role": "assistant", "content": msg.content or None}
        if raw_calls:
            raw["tool_calls"] = raw_calls
        return LLMResponse(Message(role="assistant", text=msg.content or "", tool_calls=calls, raw=raw))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OpenRouter free models offer no embeddings; use GeminiLLM for embed()")
