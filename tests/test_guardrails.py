import pytest

from agentkit import Agent, Guardrails, MCPToolset, Message, ToolCall

from conftest import DEMO, FakeLLM, say


def call(name, **args):
    return Message("assistant", tool_calls=[ToolCall(name, args)])


def run(script, guardrails, allow=("save_note", "list_notes", "add")):
    with MCPToolset(DEMO, allow=list(allow)) as mcp:
        agent = Agent(FakeLLM(script), mcp.tools(), "sys", guardrails=guardrails)
        result = agent.run("go")
        notes = agent.registry.call("list_notes", {})  # ask the real server what happened
    return result, notes


def test_read_only_flag_comes_from_the_server():
    with MCPToolset(DEMO) as mcp:
        flags = {t.name: t.read_only for t in mcp.tools()}
    assert flags["add"] is True and flags["save_note"] is False
    assert flags["delete_everything"] is None  # unannotated -> unknown -> treated as write


def test_dry_run_does_not_execute_writes_but_runs_reads():
    result, notes = run([call("save_note", text="hi"), call("add", a=1, b=2), say("ok")],
                        Guardrails("dry_run"))
    assert result.steps[1].output.startswith("dry-run:")
    assert result.steps[3].output == "3"          # reads still run
    assert notes == ""                           # the write never reached the server


def test_live_mode_executes_writes():
    _, notes = run([call("save_note", text="hi"), say("ok")], Guardrails("live"))
    assert notes == "hi"


def test_approve_mode_yes_and_no():
    _, notes = run([call("save_note", text="yes"), say("ok")],
                   Guardrails("approve", approver=lambda n, a: True))
    assert notes == "yes"
    result, notes = run([call("save_note", text="no"), say("ok")],
                        Guardrails("approve", approver=lambda n, a: False))
    assert result.steps[1].output == "error: the user denied this action"
    assert notes == ""


def test_approver_only_asked_for_writes():
    asked = []
    run([call("add", a=1, b=1), call("save_note", text="x"), say("ok")],
        Guardrails("approve", approver=lambda n, a: asked.append(n) or True))
    assert asked == ["save_note"]


def test_argument_rules_block_calls_in_every_mode():
    rules = {"save_note": lambda a: "notes may not contain 'secret'" if "secret" in a["text"] else None}
    result, notes = run([call("save_note", text="my secret"), say("ok")],
                        Guardrails("live", rules=rules))
    assert result.steps[1].output.startswith("error: blocked by policy")
    assert notes == ""


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        Guardrails("yolo")
