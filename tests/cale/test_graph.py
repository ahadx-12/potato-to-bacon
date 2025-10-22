import json
from pathlib import Path

from potatobacon.cale.graph import compute_authority_scores, load_citation_graph


def test_pagerank_and_normalization(tmp_path: Path):
    corpus = [
        {"id": "A", "cites": ["C"]},
        {"id": "B", "cites": ["C"]},
        {"id": "C", "cites": []},
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))

    graph = load_citation_graph(path)
    scores = compute_authority_scores(graph)

    assert set(scores.keys()) == {"A", "B", "C"}
    assert 0.0 <= min(scores.values()) <= max(scores.values()) <= 1.0
    assert scores["C"] > scores["A"] and scores["C"] > scores["B"]
