"""Agent loop tests: real MCP demo server, scripted (fake) LLM."""
from agentkit import Agent, MCPToolset, Message, ToolCall

from conftest import DEMO, FakeLLM, say


def call(name, **args):
    return Message("assistant", tool_calls=[ToolCall(name, args)])


def test_loop_runs_mcp_tool_then_finishes():
    llm = FakeLLM([call("add", a=2, b=3), say("done: 5")])
    with MCPToolset(DEMO, allow=["add"]) as mcp:
        result = Agent(llm, mcp.tools(), "sys").run("go")
    assert result.text == "done: 5"
    assert [s.kind for s in result.steps] == ["model", "tool", "model"]
    assert result.steps[1].output == "5"
    assert llm.seen[1][-1].results == [("add", "5")]  # tool result fed back to the model


def test_bad_arguments_are_errors_the_model_can_see():
    llm = FakeLLM([call("add", wrong=1), say("ok")])
    with MCPToolset(DEMO, allow=["add"]) as mcp:
        result = Agent(llm, mcp.tools(), "sys").run("go")
    assert result.steps[1].kind == "error"


def test_unknown_tool_is_an_error():
    llm = FakeLLM([call("nope"), say("ok")])
    with MCPToolset(DEMO, allow=["add"]) as mcp:
        result = Agent(llm, mcp.tools(), "sys").run("go")
    assert result.steps[1].output.startswith("error: unknown tool")


def test_step_limit_stops_runaway_loop():
    llm = FakeLLM([call("add", a=1, b=1) for _ in range(10)])
    with MCPToolset(DEMO, allow=["add"]) as mcp:
        result = Agent(llm, mcp.tools(), "sys", max_steps=3).run("go")
    assert result.stopped_by_limit
