from types import SimpleNamespace as NS

import pytest

from mcp_servers import triage_server as ts


def issue(number, title, body="", state="open", pr=None, labels_sink=None):
    return NS(number=number, title=title, body=body, state=state, pull_request=pr,
              add_to_labels=lambda *l: labels_sink.extend(l))


class FakeRepo:
    def __init__(self, issues):
        self.issues = {i.number: i for i in issues}

    def get_issue(self, n):
        return self.issues[n]

    def get_issues(self, state="all"):
        return list(self.issues.values())


def test_cosine():
    assert ts.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert ts.cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def test_split_labels_allows_only_the_allow_list():
    assert ts.split_labels(["bug", "delete-repo", "bug", "priority:high"]) == (
        ["bug", "priority:high"], ["delete-repo"])


def test_apply_labels_adds_only_allowed(monkeypatch):
    sink = []
    monkeypatch.setattr(ts, "_repo", lambda o, r: FakeRepo([issue(1, "x", labels_sink=sink)]))
    out = ts.apply_labels("o", "r", 1, ["bug", "evil"])
    assert sink == ["bug"]
    assert "rejected" in out and "evil" in out


def test_apply_labels_with_nothing_allowed_touches_nothing(monkeypatch):
    sink = []
    monkeypatch.setattr(ts, "_repo", lambda o, r: FakeRepo([issue(1, "x", labels_sink=sink)]))
    ts.apply_labels("o", "r", 1, ["evil"])
    assert sink == []


def test_find_similar_ranks_filters_and_skips_prs_and_self(monkeypatch):
    repo = FakeRepo([
        issue(1, "login broken"),                 # the target
        issue(2, "login fails"),                  # near duplicate
        issue(3, "add dark mode"),                # unrelated
        issue(4, "login PR", pr=object()),        # pull request: skipped
    ])
    monkeypatch.setattr(ts, "_repo", lambda o, r: repo)
    vectors = {"login broken": [1, 0], "login fails": [0.9, 0.1], "add dark mode": [0, 1]}
    monkeypatch.setattr(ts, "_embed", lambda texts: [vectors[t.split("\n")[0]] for t in texts])
    result = ts.find_similar_issues("o", "r", 1)
    assert [r["number"] for r in result] == [2]   # unrelated is under min_score
    assert result[0]["score"] > 0.9


def test_find_similar_with_no_candidates(monkeypatch):
    monkeypatch.setattr(ts, "_repo", lambda o, r: FakeRepo([issue(1, "only issue")]))
    assert ts.find_similar_issues("o", "r", 1) == []
