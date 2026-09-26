import sys

from mcp import StdioServerParameters

from agentkit.types import LLMResponse, Message


DEMO = StdioServerParameters(command=sys.executable, args=["-m", "mcp_servers.demo_server"])


class FakeLLM:
    """Replays scripted assistant messages; records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def generate(self, system, messages, tools):
        self.seen.append(list(messages))
        return LLMResponse(self.script.pop(0))

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def say(text: str) -> Message:
    return Message("assistant", text=text)
