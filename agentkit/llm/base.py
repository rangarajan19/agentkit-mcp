from __future__ import annotations

from typing import Protocol

from ..types import LLMResponse, Message, ToolSpec


class LLM(Protocol):
    """Anything that can run one model turn and embed text."""

    def generate(self, system: str, messages: list[Message],
                 tools: list[ToolSpec]) -> LLMResponse: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
