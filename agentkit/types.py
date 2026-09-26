"""Provider-neutral types shared by the agent loop, LLM adapters and tools."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    """A tool the model may invoke, backed by an MCP server.

    `parameters` is the JSON Schema the server published. `read_only` comes from the
    server's readOnlyHint annotation (None = unknown, treated as a write by guardrails).
    """
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    read_only: bool | None = None


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Message:
    """role: 'user' | 'assistant' | 'tool'.

    assistant: `text` and/or `tool_calls`; `raw` may hold the provider's original
    object so adapters can replay it exactly (e.g. Gemini thought signatures).
    tool: `results` is a list of (tool_name, output_string) answering the previous turn.
    """
    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[tuple[str, str]] = field(default_factory=list)
    raw: Any = None


@dataclass
class LLMResponse:
    message: Message

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls


@dataclass
class Step:
    """One entry in the run trace."""
    kind: str  # 'model' | 'tool' | 'error'
    name: str = ""
    args: dict = field(default_factory=dict)
    output: str = ""
    seconds: float = 0.0


@dataclass
class RunResult:
    text: str
    steps: list[Step]
    stopped_by_limit: bool = False
