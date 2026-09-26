# agentkit-mcp: an MCP-first agent framework

![tests](https://github.com/rangarajan19/agentkit-mcp/actions/workflows/tests.yml/badge.svg)

A small **agent framework** (`agentkit/`) where **every tool is an MCP tool**, with safety
guardrails at the tool layer, plus two example agents built on it. Runs on the **free tier** of
the Gemini API. No Docker needed. See [docs/architecture.md](docs/architecture.md) for the design.

## Status
Early (v0.1). What has been verified by running it:
- Agent loop, MCP client (local stdio server and the hosted GitHub MCP server), guardrails, and both
  LLM providers: covered by 47 automated tests (no network needed).
- Issue triage ran end to end in **dry-run** mode on real issues with Gemini and with the free
  `nvidia/nemotron-3-super-120b-a12b` model on OpenRouter (4 test issues: clear bug, vague report,
  duplicate, prompt-injection attempt). The research agent produced a cited report.

Not verified yet: real label/comment writes, the GitHub Actions workflow, and triage *accuracy*
(4 examples is not an evaluation set). Treat triage output as suggestions until you have run it
with `--approve`.

```
examples ──► agentkit ──► Gemini (LLM)
                │
                └─► ToolRegistry + Guardrails ──► MCP servers (stdio / HTTP)
                                                    ├─ GitHub MCP server (hosted)
                                                    ├─ our triage MCP server
                                                    └─ DuckDuckGo search MCP server
```

## Layout
```
agentkit/
  agent.py            the agent loop: model -> tool calls -> observations -> repeat
  llm/base.py         LLM interface (generate, embed)
  llm/gemini.py       Gemini adapter: retry, and fallback across a model chain
  llm/openrouter.py   OpenRouter adapter (OpenAI-compatible; many free models)
  llm/factory.py      make_llm("openrouter:<model>" | "gemini:<a>,<b>")
  tools/mcp_client.py MCPToolset: connect to an MCP server, expose its tools (allow/deny lists)
  tools/registry.py   one lookup table of tools; the single place calls are dispatched
  guardrails.py       dry-run / human approval / per-tool argument rules
  types.py            provider-neutral Message / ToolSpec / Step
mcp_servers/
  triage_server.py    our MCP server: find_similar_issues, apply_labels (allow-listed, add-only)
  demo_server.py      tiny server used by tests
examples/
  issue_triage/       labels, dedupes and replies to GitHub issues
  research_agent/     web search + fetch -> cited markdown report
```

## How a tool call works
1. `MCPToolset` connects to an MCP server and asks `tools/list`. The server's own name,
   description and JSON schema become the agent's `ToolSpec`s (nothing is hand-written).
2. The model answers with "call `add(a=1, b=2)`". The agent loop hands that to the registry.
3. **Guardrails** run first: argument rules, then dry-run / approval for write tools.
4. The registry forwards the call to the MCP server (`tools/call`) and returns the text result
   to the model. Errors come back as text so the model can correct itself.

Servers that annotate tools with `readOnlyHint` (GitHub's does) let guardrails tell reads from
writes automatically. Unannotated tools count as writes unless you vouch with `trusted_read_only`.

## Use it
```python
import sys
from mcp import StdioServerParameters
from agentkit import Agent, GeminiLLM, Guardrails, MCPToolset

server = StdioServerParameters(command=sys.executable, args=["-m", "mcp_servers.demo_server"])
with MCPToolset(server, allow=["add", "word_count"]) as mcp:   # a URL string works too
    agent = Agent(GeminiLLM(), mcp.tools(), "Use tools for math.", guardrails=Guardrails("dry_run"))
    result = agent.run("What is 1234 + 8766?")
    print(result.text)          # result.steps holds the full trace
```
Guardrail modes: `live`, `dry_run` (writes are reported, never executed), `approve` (ask a human).

## Choosing a model
Default is a Gemini chain (`gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash`): if a model is out of
free quota or overloaded, the next one is used. Free models on [OpenRouter](https://openrouter.ai/models)
also work (set `OPENROUTER_API_KEY`; free models need a `:free` suffix and are rate limited):
```
python -m examples.issue_triage.main --repo o/r --issue 1 --model "openrouter:nvidia/nemotron-3-super-120b-a12b:free"
```
Free tiers change often, so check current limits. OpenRouter has no free embedding models, so duplicate
detection keeps using Gemini embeddings.

## Example 1: issue triage agent
```
python -m examples.issue_triage.main --repo owner/repo --issue 1            # dry run (default)
python -m examples.issue_triage.main --repo owner/repo --issue 1 --approve  # confirm each write
```
Tools: GitHub MCP `issue_read`, `add_issue_comment`, `get_file_contents` (3 of its 45 tools,
allow-listed) + our `find_similar_issues` and `apply_labels`. Safety layers: tool allow-list,
target repo/issue scoping, comment length cap, server-side label allow-list, dry-run/approval, step limit.
The `duplicate` label is enforced in code: it requires `duplicate_of` pointing at an *older* issue, so the
original report can never be marked as a copy of a newer one.
Issue text is treated as untrusted. Set `DRY_RUN=false` in `.env` to write for real.

## Example 2: research agent
```
python -m examples.research_agent.main "topic" --out report.md
```
Uses the `duckduckgo-mcp-server` (started with `uvx`, no API key) for search and page fetch.

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env        # GEMINI_API_KEY (free), GITHUB_TOKEN + GITHUB_REPOSITORY for triage
pytest
```
The research agent also needs [`uv`](https://docs.astral.sh/uv/) (for `uvx`).

### Run triage automatically on new issues (opt-in)
The workflow in `.github/workflows/triage.yml` is off by default. To enable it:
1. Add repository secrets `GEMINI_API_KEY` and `TRIAGE_GITHUB_PAT` (a fine-grained token with
   Issues read/write on the repo). The hosted GitHub MCP server needs a personal token, not the
   built-in Actions `GITHUB_TOKEN`.
2. Add a repository variable `ENABLE_TRIAGE` = `true` (Settings > Secrets and variables > Actions > Variables).

The workflow writes for real (`DRY_RUN=false`). Try `--approve` locally first.

## Notes
- Free-tier Gemini content may be used by Google to improve its products: use public data only.
- Rate limits are per project (aistudio.google.com/rate-limit); 429s are retried with backoff.
- MCP tool output is untrusted input, so keep `allow` lists tight.

## Contributing and license
See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under the [MIT License](LICENSE).

## Roadmap
- [x] Framework core, MCP client, guardrails, tracing
- [x] GitHub MCP server integration, our own MCP server, two example agents
- [x] Second LLM provider (OpenRouter) and model fallback
- [ ] Evaluation set to measure triage accuracy across models
- [ ] More providers (Groq, Ollama)
- [ ] Persist conversation memory across runs
- [ ] Demo GIF and architecture diagram
