"""MCP client: expose any MCP server's tools as ordinary agentkit ToolSpecs.

The agent loop is synchronous while MCP is async, so each MCPToolset owns a small
background event loop. One long-lived task opens the connection, keeps it alive,
and closes it (anyio requires open/close to happen in the same task).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from ..types import ToolSpec


def _strip_titles(schema):
    """Drop pydantic's noisy 'title' keys (not needed by the model, some providers dislike them)."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items()
                if not (k == "title" and isinstance(v, str))}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


class MCPToolset:
    """Connect to one MCP server and turn its tools into ToolSpecs.

    server: StdioServerParameters (launch a local process) or a URL string (streamable HTTP).
    headers: HTTP headers for a URL server, e.g. {"Authorization": "Bearer <token>"}.
    allow:  if set, only these tool names are exposed (recommended for powerful servers).
    deny:   tool names to hide.
    prefix: prepended to tool names, to avoid collisions between servers.
    trusted_read_only: tool names YOU vouch for as read-only, for servers that don't
        annotate their tools (unannotated tools are otherwise treated as writes by guardrails).
    """

    def __init__(self, server: StdioServerParameters | str, *, allow: list[str] | None = None,
                 deny: list[str] | None = None, prefix: str = "", timeout: float = 60.0,
                 headers: dict[str, str] | None = None,
                 trusted_read_only: list[str] | None = None):
        self.server = server
        self.trusted_read_only = set(trusted_read_only or [])
        self.headers = headers
        self.allow = set(allow) if allow is not None else None
        self.deny = set(deny or [])
        self.prefix = prefix
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._stop: asyncio.Event | None = None
        self._main_future: concurrent.futures.Future | None = None
        self._client: Client | None = None
        self._specs: list[ToolSpec] = []

    # --- lifecycle ----------------------------------------------------------
    async def _main(self):
        self._stop = asyncio.Event()
        try:
            async with Client(self._transport()) as client:
                self._client = client
                listed = (await client.list_tools()).tools
                self._specs = [self._to_spec(t) for t in listed if self._visible(t.name)]
                self._ready.set_result(None)
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - surface startup failures to start()
            if not self._ready.done():
                self._ready.set_exception(e)
            else:
                raise

    def _transport(self):
        if isinstance(self.server, str) and self.headers:
            http = create_mcp_http_client(headers=self.headers)
            return streamable_http_client(self.server, http_client=http)
        return self.server

    def start(self) -> "MCPToolset":
        self._thread.start()
        self._main_future = asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        self._ready.result(timeout=self.timeout)
        return self

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._main_future is not None:
            try:
                self._main_future.result(timeout=10)
            except Exception:  # noqa: BLE001 - best-effort shutdown
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> "MCPToolset":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- tools --------------------------------------------------------------
    def _visible(self, name: str) -> bool:
        return (self.allow is None or name in self.allow) and name not in self.deny

    def _to_spec(self, t) -> ToolSpec:
        name = t.name

        def call(**kwargs) -> str:
            return self._call(name, kwargs)

        return ToolSpec(
            name=f"{self.prefix}{name}",
            description=(t.description or "").strip(),
            parameters=_strip_titles(t.input_schema) or {"type": "object", "properties": {}},
            fn=call,
            read_only=True if name in self.trusted_read_only else (
                getattr(t.annotations, "read_only_hint", None) if t.annotations else None),
        )

    def _call(self, name: str, args: dict) -> str:
        fut = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, args), self._loop)
        result = fut.result(timeout=self.timeout)
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        return f"error: {text}" if result.is_error else text

    def tools(self) -> list[ToolSpec]:
        return list(self._specs)
