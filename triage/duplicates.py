import numpy as np

from . import llm


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_similar(query: str, issues: list[dict], top_k: int = 3) -> list[dict]:
    """issues: [{"number", "title", "body"}]. Returns best matches with a score."""
    if not issues:
        return []
    texts = [query] + [f"{i['title']}\n{(i['body'] or '')[:1000]}" for i in issues]
    vecs = llm.embed(texts)
    scored = [
        {"number": i["number"], "title": i["title"], "score": round(cosine(vecs[0], v), 3)}
        for i, v in zip(issues, vecs[1:])
    ]
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored[:top_k]
