import json
from types import SimpleNamespace as NS

import pytest

from agentkit import Message, OpenRouterLLM, ToolCall, ToolSpec, make_llm
from agentkit.llm import openrouter as orr
from agentkit.llm.gemini import GeminiLLM


class HTTPish(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status_code = status


def completion(content=None, calls=()):
    tcs = [NS(id=f"call_{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(calls)]
    return NS(choices=[NS(message=NS(content=content, tool_calls=tcs or None))])


def llm_returning(*responses):
    """OpenRouterLLM whose HTTP client replays `responses` (exceptions are raised)."""
    llm = OpenRouterLLM(model="m", api_key="x")
    seq, sent = list(responses), []

    def create(**kwargs):
        sent.append(kwargs)
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    llm.client = NS(chat=NS(completions=NS(create=create)))
    return llm, sent


TOOL = ToolSpec("add", "Add.", {"type": "object", "properties": {"a": {"type": "integer"}}}, fn=lambda **k: "")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(orr.time, "sleep", lambda s: None)


def test_tool_call_is_parsed_and_tools_are_sent_in_openai_format():
    llm, sent = llm_returning(completion(calls=[("add", '{"a": 2}')]))
    r = llm.generate("sys", [Message("user", text="go")], [TOOL])
    assert r.tool_calls == [ToolCall("add", {"a": 2})]
    assert sent[0]["tools"][0]["function"]["name"] == "add"
    assert sent[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_tool_results_are_linked_to_call_ids_in_order():
    llm, sent = llm_returning(completion(calls=[("add", '{"a": 1}'), ("add", '{"a": 2}')]), completion("done"))
    first = llm.generate("s", [Message("user", text="go")], [TOOL])
    history = [Message("user", text="go"), first.message,
               Message("tool", results=[("add", "R1"), ("add", "R2")])]
    llm.generate("s", history, [TOOL])
    tool_msgs = [m for m in sent[1]["messages"] if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [("call_0", "R1"), ("call_1", "R2")]


def test_malformed_json_arguments_do_not_crash():
    llm, _ = llm_returning(completion(calls=[("add", "{not json")]))
    r = llm.generate("s", [Message("user", text="go")], [TOOL])
    assert "_invalid_json" in r.tool_calls[0].args


def test_temporary_errors_are_retried():
    llm, sent = llm_returning(HTTPish(429), HTTPish(503), completion("ok"))
    assert llm.generate("s", [Message("user", text="go")], []).text == "ok"
    assert len(sent) == 3


def test_permanent_errors_are_not_retried():
    llm, sent = llm_returning(HTTPish(401), completion("ok"))
    with pytest.raises(HTTPish):
        llm.generate("s", [Message("user", text="go")], [])
    assert len(sent) == 1


def test_embed_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        OpenRouterLLM(model="m", api_key="x").embed(["x"])


def test_make_llm_specs():
    assert isinstance(make_llm("openrouter:nvidia/x:free"), OpenRouterLLM)
    assert make_llm("openrouter:nvidia/x:free").model == "nvidia/x:free"
    assert make_llm("gemini:a,b").models == ["a", "b"]
    assert isinstance(make_llm(None), GeminiLLM)


def test_error_payload_inside_a_200_response_is_retried():
    overloaded = NS(choices=[], error={"code": 503, "message": "Service temporarily overloaded"})
    llm, sent = llm_returning(overloaded, completion("ok"))
    assert llm.generate("s", [Message("user", text="go")], []).text == "ok"
    assert len(sent) == 2


def test_error_payload_with_permanent_code_is_raised():
    bad = NS(choices=[], error={"code": 401, "message": "bad key"})
    llm, sent = llm_returning(bad, completion("ok"))
    with pytest.raises(orr.ProviderError):
        llm.generate("s", [Message("user", text="go")], [])
    assert len(sent) == 1
