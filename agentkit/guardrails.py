"""Tool-layer safety. Applied to every tool call before it reaches an MCP server.

Modes:
  live     run everything (rules still apply)
  dry_run  read-only tools run; write tools are NOT executed, the call is only reported
  approve  read-only tools run; each write tool call must be approved by a human

`rules` maps a tool name to a function(args) -> error string or None, for argument-level
policy such as "only these labels" or "never set state=closed".
"""
from __future__ import annotations

from typing import Callable

from .types import ToolSpec

Rule = Callable[[dict], "str | None"]
Approver = Callable[[str, dict], bool]


def cli_approver(name: str, args: dict) -> bool:
    print(f"\nApprove tool call?  {name}({args})")
    return input("[y/N] ").strip().lower() in ("y", "yes")


class Guardrails:
    def __init__(self, mode: str = "live", approver: Approver | None = None,
                 rules: dict[str, Rule] | None = None):
        if mode not in ("live", "dry_run", "approve"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.approver = approver or cli_approver
        self.rules = rules or {}

    @staticmethod
    def is_write(spec: ToolSpec) -> bool:
        return spec.read_only is not True  # unknown counts as a write

    def check(self, spec: ToolSpec, args: dict) -> str | None:
        """Return text to use instead of running the tool, or None to run it."""
        rule = self.rules.get(spec.name)
        if rule:
            problem = rule(args)
            if problem:
                return f"error: blocked by policy: {problem}"
        if not self.is_write(spec):
            return None
        if self.mode == "dry_run":
            return f"dry-run: {spec.name} was NOT executed (would run with {args})"
        if self.mode == "approve" and not self.approver(spec.name, args):
            return "error: the user denied this action"
        return None
