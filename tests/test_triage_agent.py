from examples.issue_triage.agent import scoped_rules

rules = scoped_rules("me", "proj", 5)


def test_other_repo_or_issue_is_blocked():
    assert rules["issue_read"]({"owner": "me", "repo": "proj", "issue_number": 5}) is None
    assert "only me/proj" in rules["issue_read"]({"owner": "evil", "repo": "proj", "issue_number": 5})
    assert "only issue #5" in rules["apply_labels"]({"owner": "me", "repo": "proj", "issue_number": 6})


def test_issue_number_as_float_is_accepted():  # models often send 5.0 for a JSON "number"
    assert rules["add_issue_comment"]({"owner": "me", "repo": "proj", "issue_number": 5.0, "body": "hi"}) is None


def test_comment_rules():
    base = {"owner": "me", "repo": "proj", "issue_number": 5}
    assert "empty" in rules["add_issue_comment"]({**base, "body": "  "})
    assert "longer" in rules["add_issue_comment"]({**base, "body": "x" * 5000})
    assert "empty" in rules["add_issue_comment"](base)  # reaction-only calls are rejected
