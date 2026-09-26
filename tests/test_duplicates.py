from triage import duplicates


def test_cosine_identical_is_one():
    assert abs(duplicates.cosine([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(duplicates.cosine([1, 0], [0, 1])) < 1e-9


def test_find_similar_ranks(monkeypatch):
    monkeypatch.setattr(duplicates.llm, "embed", lambda texts: [[1, 0], [0, 1], [1, 0.1]])
    issues = [{"number": 1, "title": "a", "body": ""}, {"number": 2, "title": "b", "body": ""}]
    assert duplicates.find_similar("q", issues)[0]["number"] == 2
