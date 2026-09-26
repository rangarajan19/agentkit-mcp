"""The registry: one lookup table of every tool the agent may use, plus the single
place where tool calls are dispatched (so guardrails apply to all of them)."""
from __future__ import annotations

import json

from ..guardrails import Guardrails
from ..types import ToolSpec


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec] | None = None,
                 guardrails: Guardrails | None = None):
        self._tools: dict[str, ToolSpec] = {}
        self.guardrails = guardrails or Guardrails("live")
        for t in tools or []:
            self.add(t)

    def add(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools[spec.name] = spec

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def call(self, name: str, args: dict) -> str:
        """Run a tool; errors come back as text so the model can recover."""
        spec = self._tools.get(name)
        if spec is None:
            return f"error: unknown tool '{name}'"
        blocked = self.guardrails.check(spec, args)
        if blocked is not None:
            return blocked
        try:
            out = spec.fn(**args)
            return out if isinstance(out, str) else json.dumps(out, default=str)
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"
