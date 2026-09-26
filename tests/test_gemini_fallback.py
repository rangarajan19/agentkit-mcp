from types import SimpleNamespace as NS

import pytest

from agentkit import GeminiLLM, Message
from agentkit.llm import gemini as g


class Boom(Exception):
    def __init__(self, code, msg=""):
        super().__init__(f"{code} {msg}")
        self.code = code


def fake_client(behaviour):
    """behaviour: {model: exception | 'ok'}; records which models were called."""
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        b = behaviour[model]
        if isinstance(b, Exception):
            raise b
        part = NS(text="hello", function_call=None, thought=False)
        return NS(candidates=[NS(content=NS(parts=[part]))])

    return NS(models=NS(generate_content=generate_content)), calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)


def make(models, behaviour):
    llm = GeminiLLM(api_key="x", model=models)
    llm.client, calls = fake_client(behaviour)
    return llm, calls


def ask(llm):
    return llm.generate("sys", [Message("user", text="hi")], []).text


def test_daily_quota_falls_back_without_retrying_the_same_model():
    llm, calls = make("a,b", {"a": Boom(429, "PerDay quota"), "b": "ok"})
    assert ask(llm) == "hello"
    assert calls == ["a", "b"]


def test_temporary_overload_is_retried_then_falls_back():
    llm, calls = make("a,b", {"a": Boom(503, "high demand"), "b": "ok"})
    assert ask(llm) == "hello"
    assert calls == ["a"] * 4 + ["b"]          # 4 attempts on a, then b


def test_non_capacity_errors_are_not_swallowed():
    llm, calls = make("a,b", {"a": Boom(400, "bad request"), "b": "ok"})
    with pytest.raises(Boom):
        ask(llm)
    assert calls == ["a"]


def test_all_models_failing_raises_the_last_error():
    llm, _ = make("a,b", {"a": Boom(503), "b": Boom(503)})
    with pytest.raises(Boom):
        ask(llm)


def test_single_model_string_still_works():
    llm, calls = make("only", {"only": "ok"})
    assert ask(llm) == "hello" and calls == ["only"]


def test_fallback_is_sticky_so_dead_models_are_not_retried_every_call():
    llm, calls = make("a,b", {"a": Boom(429, "PerDay quota"), "b": "ok"})
    ask(llm)
    ask(llm)
    ask(llm)
    assert calls == ["a", "b", "b", "b"]   # 'a' is tried once, then skipped
    assert llm.model == "b"
