# Architecture

## Big picture
```
examples/ (issue_triage, research_agent)
      │  Agent(llm, tools, system_prompt, guardrails).run(prompt)
      ▼
┌─────────────────────────────── agentkit ────────────────────────────────┐
│                                                                         │
│  Agent loop ──messages + tool specs──►  LLM interface ──► Gemini API    │
│  (agent.py)  ◄──text and/or tool calls──  (llm/base.py, llm/gemini.py)  │
│      │                                                                  │
│      │ call(name, args)                                                 │
│      ▼                                                                  │
│  ToolRegistry ── Guardrails (rules → dry-run / approval) ──┐            │
│  (tools/registry.py)   (guardrails.py)                     │            │
│                                                            ▼            │
│                                    MCPToolset (tools/mcp_client.py)     │
└────────────────────────────────────────────┬────────────────────────────┘
                                             │ MCP protocol (stdio or HTTP)
              ┌──────────────────────────────┼───────────────────────────┐
              ▼                              ▼                           ▼
     GitHub MCP server (hosted)   our triage MCP server      DuckDuckGo MCP server
```

## The agent loop
1. The user prompt becomes the first message.
2. `llm.generate(system, messages, tools)` returns text, tool calls, or both.
3. No tool calls: return the text as the final answer.
4. Tool calls: each goes through `ToolRegistry.call()`, which applies guardrails and then
   forwards the call to the right MCP server. Results are appended as a `tool` message.
5. Repeat until the model answers or `max_steps` is reached.

Every model turn and tool call is recorded as a `Step` (the trace in `RunResult.steps`).

## Components
| Piece | Job |
|---|---|
| `types.py` | Provider-neutral `Message`, `ToolSpec`, `Step`, `RunResult` |
| `llm/base.py` | `LLM` interface: `generate` and `embed`. Add a provider by implementing it |
| `llm/gemini.py` | Converts to/from Gemini's format; retries on 429 (free-tier rate limits) |
| `tools/mcp_client.py` | `MCPToolset`: connects to one MCP server and turns its tools into `ToolSpec`s. Filters: `allow`, `deny`, `prefix`, `trusted_read_only` |
| `tools/registry.py` | One lookup table of tools; the single dispatch point, so guardrails cover every call |
| `guardrails.py` | Modes `live` / `dry_run` / `approve`, plus per-tool argument rules |
| `mcp_servers/triage_server.py` | Our MCP server: `find_similar_issues`, `apply_labels` (allow-listed, add-only) |

## Design decisions
- **MCP-only tools.** There is no local tool decorator. A tool is an MCP tool, so the same tools
  work in other MCP clients and the schema always comes from the server.
- **Own loop, not the SDK's automatic function calling**, so we control step limits, tracing,
  error handling and approvals.
- **Sync loop, async MCP.** Each `MCPToolset` runs a small event loop in a background thread and
  keeps its server connection open for the life of the toolset.
- **Read/write awareness.** Servers annotate tools with `readOnlyHint`. Guardrails treat
  unannotated tools as writes unless you vouch for them with `trusted_read_only`.

## Threat model (short)
Inputs from issues and web pages are untrusted and may contain prompt injection. Defence is at the
tool layer, not in the prompt alone:
1. Tool allow-lists (the agent sees 3 of the GitHub server's 45 tools).
2. Argument rules (target repo and issue only, comment length cap).
3. Narrow write tools (our `apply_labels` can only add labels from a fixed list).
4. Dry-run or human approval for every write, and a step limit on the loop.

These reduce the damage of a successful injection; they do not prove injections cannot happen.
