"""Authority graph utilities for CALE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np


def load_citation_graph(path: str | Path) -> nx.DiGraph:
    dataset_path = Path(path)
    data = json.loads(dataset_path.read_text())
    graph = nx.DiGraph()
    for item in data:
        rid = item["id"]
        graph.add_node(
            rid,
            **{k: item.get(k) for k in ("jurisdiction", "statute", "section", "year")},
        )
    for item in data:
        for target in item.get("cites", []):
            if graph.has_node(item["id"]) and graph.has_node(target):
                graph.add_edge(item["id"], target)
    return graph


def compute_authority_scores(graph: nx.DiGraph, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}

    nodes: List[str] = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    transition = np.zeros((n, n), dtype=np.float64)

    for source in nodes:
        i = index[source]
        successors = [target for target in graph.successors(source) if target in index]
        if successors:
            weight = 1.0 / len(successors)
            for target in successors:
                j = index[target]
                transition[j, i] = weight
        else:
            transition[:, i] = 1.0 / n

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(max_iter):
        prev = rank.copy()
        rank = alpha * transition @ prev + (1 - alpha) * teleport
        if np.linalg.norm(rank - prev, 1) < tol:
            break

    if rank.sum() > 0:
        rank /= rank.sum()

    lo, hi = float(rank.min()), float(rank.max())
    if hi - lo <= 1e-12:
        return {node: 0.0 for node in nodes}
    return {node: (float(rank[index[node]]) - lo) / (hi - lo) for node in nodes}


__all__ = ["load_citation_graph", "compute_authority_scores"]
