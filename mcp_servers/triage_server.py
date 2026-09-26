"""Triage MCP server: narrow, safe tools for issue triage.

Deliberately small: the broad GitHub MCP server can close, edit and delete things;
this one can only (a) rank similar issues and (b) ADD labels from an allow-list.

Env: GITHUB_TOKEN, GEMINI_API_KEY.  Run standalone: python -m mcp_servers.triage_server
"""
from __future__ import annotations

import os

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

ALLOWED_LABELS = {
    "bug", "enhancement", "question", "documentation", "duplicate",
    "needs-info", "priority:high", "priority:medium", "priority:low",
}
MAX_CANDIDATES = 100   # issues compared against
MAX_SCANNED = 150      # raw items examined (PRs are mixed in with issues)

server = MCPServer("triage")


# --- backends (module-level so tests can replace them) -------------------------
def _repo(owner: str, repo: str):
    from github import Auth, Github
    gh = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]), per_page=100)
    return gh.get_repo(f"{owner}/{repo}")


def _embed(texts: list[str]) -> list[list[float]]:
    from agentkit.llm.gemini import GeminiLLM
    return GeminiLLM().embed(texts)


# --- pure helpers --------------------------------------------------------------
def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def split_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    """(accepted, rejected). Order kept, duplicates dropped."""
    accepted, rejected = [], []
    for label in labels:
        bucket = accepted if label in ALLOWED_LABELS else rejected
        if label not in bucket:
            bucket.append(label)
    return accepted, rejected


def _text(title: str, body: str | None) -> str:
    return f"{title}\n{(body or '')[:1000]}"


# --- tools ---------------------------------------------------------------------
@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def find_similar_issues(owner: str, repo: str, issue_number: int,
                        top_k: int = 3, min_score: float = 0.6) -> list[dict]:
    """Find existing issues in the repo that are semantically similar to the given issue
    (possible duplicates). Returns number, title, state, a 0-1 similarity score, and whether the
    match is older than this issue (only an OLDER issue can be the original)."""
    r = _repo(owner, repo)
    target = r.get_issue(issue_number)
    candidates = []
    for scanned, i in enumerate(r.get_issues(state="all"), start=1):
        if not (i.pull_request or i.number == issue_number):
            candidates.append(i)
        if len(candidates) >= MAX_CANDIDATES or scanned >= MAX_SCANNED:
            break
    if not candidates:
        return []
    vecs = _embed([_text(target.title, target.body)] + [_text(i.title, i.body) for i in candidates])
    scored = [{"number": i.number, "title": i.title, "state": i.state,
               "older_than_this_issue": i.number < issue_number,
               "score": round(cosine(vecs[0], v), 3)} for i, v in zip(candidates, vecs[1:])]
    scored = [s for s in scored if s["score"] >= min_score]
    return sorted(scored, key=lambda s: s["score"], reverse=True)[:top_k]


def check_duplicate(labels: list[str], issue_number: int, duplicate_of: int | None) -> tuple[list[str], str]:
    """The 'duplicate' label is only valid when pointing at an OLDER issue. Otherwise drop it,
    so the original report is never marked as the duplicate of a newer copy."""
    if "duplicate" not in labels:
        return labels, ""
    if duplicate_of is None or duplicate_of >= issue_number:
        kept = [l for l in labels if l != "duplicate"]
        return kept, ("'duplicate' dropped: it needs duplicate_of set to an OLDER issue number "
                      f"(got {duplicate_of}, this issue is #{issue_number})")
    return labels, ""


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
def apply_labels(owner: str, repo: str, issue_number: int, labels: list[str],
                 duplicate_of: int | None = None) -> str:
    """Add labels to an issue. Only these labels are allowed: bug, enhancement, question,
    documentation, duplicate, needs-info, priority:high, priority:medium, priority:low.
    The 'duplicate' label requires duplicate_of = the number of an OLDER issue that is the original.
    Labels are only ever added, never removed."""
    labels, note = check_duplicate(labels, issue_number, duplicate_of)
    accepted, rejected = split_labels(labels)
    if accepted:
        _repo(owner, repo).get_issue(issue_number).add_to_labels(*accepted)
    msg = f"applied: {accepted}"
    if rejected:
        msg += f"; rejected (not allowed): {rejected}"
    return msg + (f"; {note}" if note else "")


if __name__ == "__main__":
    server.run("stdio")
