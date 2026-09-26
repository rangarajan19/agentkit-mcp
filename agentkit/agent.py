"""The agent loop: model -> tool calls -> observations -> model, until done."""
from __future__ import annotations

import time

from .guardrails import Guardrails
from .llm.base import LLM
from .tools.registry import ToolRegistry
from .types import Message, RunResult, Step, ToolSpec


class Agent:
    def __init__(self, llm: LLM, tools: list[ToolSpec], system: str, max_steps: int = 8,
                 guardrails: Guardrails | None = None):
        self.llm = llm
        self.registry = ToolRegistry(tools, guardrails)
        self.system = system
        self.max_steps = max_steps

    def run(self, prompt: str) -> RunResult:
        messages = [Message(role="user", text=prompt)]
        steps: list[Step] = []

        for _ in range(self.max_steps):
            t0 = time.time()
            response = self.llm.generate(self.system, messages, self.registry.specs())
            steps.append(Step("model", output=response.text, seconds=time.time() - t0))
            messages.append(response.message)

            if not response.tool_calls:
                return RunResult(text=response.text, steps=steps)

            results = []
            for call in response.tool_calls:
                t0 = time.time()
                output = self.registry.call(call.name, call.args)
                kind = "error" if output.startswith("error:") else "tool"
                steps.append(Step(kind, call.name, call.args, output, time.time() - t0))
                results.append((call.name, output))
            messages.append(Message(role="tool", results=results))

        return RunResult(text="(stopped: step limit reached)", steps=steps, stopped_by_limit=True)
