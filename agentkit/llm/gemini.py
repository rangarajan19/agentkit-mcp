"""Gemini adapter. All provider-specific code lives here."""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ..types import LLMResponse, Message, ToolCall, ToolSpec

load_dotenv()


DEFAULT_MODELS = "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash"
CAPACITY_CODES = (429, 500, 503)


def is_capacity_error(e: Exception) -> bool:
    """Quota, rate-limit or overload errors: another attempt or model may succeed."""
    return getattr(e, "code", None) in CAPACITY_CODES


def _retry(fn, attempts: int = 4):
    """Retry temporary errors (rate limit, overload) with exponential backoff.
    A per-DAY quota error will not clear by waiting, so it is raised immediately."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - SDK raises several error types
            if not is_capacity_error(e) or "PerDay" in str(e) or i == attempts - 1:
                raise
            time.sleep(2 ** i)


class GeminiLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 embed_model: str | None = None):
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY", ""))
        # "a,b,c": try model a; if it is out of quota or overloaded, fall back to b, then c
        spec = model or os.getenv("GEMINI_MODEL", DEFAULT_MODELS)
        self.models = [m.strip() for m in spec.split(",") if m.strip()]
        self._active = 0  # index of the model in use; moves forward when one is unavailable
        self.embed_model = embed_model or os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

    # --- conversion helpers -------------------------------------------------
    @staticmethod
    def _declarations(tools: list[ToolSpec]) -> list[types.Tool]:
        if not tools:
            return []
        decls = [types.FunctionDeclaration(
            name=t.name, description=t.description,
            parameters_json_schema=t.parameters) for t in tools]
        return [types.Tool(function_declarations=decls)]

    @staticmethod
    def _contents(messages: list[Message]) -> list[types.Content]:
        out = []
        for m in messages:
            if m.role == "user":
                out.append(types.Content(role="user", parts=[types.Part(text=m.text)]))
            elif m.role == "assistant":
                if m.raw is not None:  # replay original content (keeps thought signatures)
                    out.append(m.raw)
                    continue
                parts = [types.Part(text=m.text)] if m.text else []
                parts += [types.Part.from_function_call(name=c.name, args=c.args)
                          for c in m.tool_calls]
                out.append(types.Content(role="model", parts=parts))
            elif m.role == "tool":
                out.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(name=n, response={"result": r})
                    for n, r in m.results]))
        return out

    def _generate_with_fallback(self, system, messages, tools):
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=self._declarations(tools) or None,
            temperature=0.2,
            # we run the loop ourselves, so turn off the SDK's automatic loop
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = self._contents(messages)
        for i in range(self._active, len(self.models)):
            model = self.models[i]
            try:
                res = _retry(lambda: self.client.models.generate_content(
                    model=model, contents=contents, config=config))
                self._active = i  # stick with a model that works instead of retrying dead ones
                return res
            except Exception as e:  # noqa: BLE001
                if i == len(self.models) - 1 or not is_capacity_error(e):
                    raise
                print(f"[agentkit] {model} unavailable ({getattr(e, 'code', '?')}); "
                      f"falling back to {self.models[i + 1]}", file=sys.stderr)

    @property
    def model(self) -> str:
        return self.models[self._active]

    # --- LLM protocol -------------------------------------------------------
    def generate(self, system: str, messages: list[Message],
                 tools: list[ToolSpec]) -> LLMResponse:
        res = self._generate_with_fallback(system, messages, tools)
        content = res.candidates[0].content if res.candidates else None
        parts = (content.parts or []) if content else []
        text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
        calls = [ToolCall(name=p.function_call.name, args=dict(p.function_call.args or {}))
                 for p in parts if p.function_call]
        return LLMResponse(Message(role="assistant", text=text, tool_calls=calls, raw=content))

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = _retry(lambda: self.client.models.embed_content(
            model=self.embed_model, contents=texts))
        return [e.values for e in res.embeddings]
