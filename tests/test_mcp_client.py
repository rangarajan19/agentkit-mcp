import sys

import pytest
from mcp import StdioServerParameters

from agentkit import MCPToolset, ToolRegistry
from agentkit.tools.mcp_client import _strip_titles

from conftest import DEMO


def test_strip_titles_keeps_property_named_title():
    schema = {"title": "X", "properties": {"title": {"title": "Title", "type": "string"}}}
    assert _strip_titles(schema) == {"properties": {"title": {"type": "string"}}}


def test_discovers_tools_and_calls_them():
    with MCPToolset(DEMO) as mcp:
        names = {t.name for t in mcp.tools()}
        assert {"add", "word_count", "delete_everything"} <= names
        reg = ToolRegistry(mcp.tools())
        assert reg.call("add", {"a": 2, "b": 3}) == "5"
        assert reg.call("word_count", {"text": "a b c"}) == "3"


def test_schema_comes_from_the_server():
    with MCPToolset(DEMO, allow=["add"]) as mcp:
        spec = mcp.tools()[0]
    assert spec.description == "Add two integers."
    assert spec.parameters["properties"]["a"] == {"type": "integer"}
    assert spec.parameters["required"] == ["a", "b"]


def test_allow_list_hides_dangerous_tools():
    with MCPToolset(DEMO, allow=["add"]) as mcp:
        assert [t.name for t in mcp.tools()] == ["add"]


def test_deny_list_and_prefix():
    with MCPToolset(DEMO, deny=["delete_everything", "save_note", "list_notes"], prefix="demo_") as mcp:
        assert {t.name for t in mcp.tools()} == {"demo_add", "demo_word_count"}


def test_server_side_errors_come_back_as_error_text():
    with MCPToolset(DEMO) as mcp:
        out = ToolRegistry(mcp.tools()).call("add", {"a": "x"})
        assert out.startswith("error:")


def test_bad_server_fails_fast():
    bad = StdioServerParameters(command=sys.executable, args=["-c", "raise SystemExit(1)"])
    with pytest.raises(Exception):
        MCPToolset(bad, timeout=15).start()


def test_trusted_read_only_overrides_unannotated_tools():
    with MCPToolset(DEMO, allow=["delete_everything", "add"], trusted_read_only=["delete_everything"]) as mcp:
        flags = {t.name: t.read_only for t in mcp.tools()}
    assert flags == {"delete_everything": True, "add": True}
