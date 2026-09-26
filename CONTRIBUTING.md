# Contributing

Thanks for looking. Small, focused pull requests are easiest to review.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate    # Windows (use source .venv/bin/activate elsewhere)
pip install -r requirements.txt
pytest
```
Tests use a scripted fake LLM and a local demo MCP server, so they need no API keys or network.

## Guidelines
- Keep provider-specific code inside `agentkit/llm/<provider>.py`.
- Tools come from MCP servers only. To add a capability, write or connect an MCP server.
- Anything that writes to the outside world must be annotated `read_only_hint=False` so guardrails apply.
- Add a test for every behaviour change. Guardrail changes need a test that proves the write did not happen.

## Ideas that would help
- More LLM providers (Groq, Ollama) behind the `LLM` interface
- An evaluation set for triage accuracy
- Memory across runs
