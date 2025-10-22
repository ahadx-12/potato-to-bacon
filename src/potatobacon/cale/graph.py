"""Authority graph utilities for CALE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional dependency
    import networkx as nx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - lightweight fallback graph
    class _DiGraph:
        def __init__(self) -> None:
            self._nodes: Dict[str, Dict[str, object]] = {}
            self._edges: Dict[str, set[str]] = {}

        def add_node(self, node: str, **attrs: object) -> None:
            self._nodes[node] = dict(attrs)
            self._edges.setdefault(node, set())

        def add_edge(self, source: str, target: str) -> None:
            if source in self._nodes and target in self._nodes:
                self._edges.setdefault(source, set()).add(target)

        def has_node(self, node: str) -> bool:
            return node in self._nodes

        def number_of_nodes(self) -> int:
            return len(self._nodes)

        def nodes(self) -> List[str]:
            return list(self._nodes.keys())

        def successors(self, node: str) -> List[str]:
            return list(self._edges.get(node, set()))

    class _NetworkXNamespace:
        DiGraph = _DiGraph

    nx = _NetworkXNamespace()  # type: ignore[assignment]

import numpy as np

import os
import random

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - maintain deterministic fallback
    torch = None  # type: ignore[assignment]

SEED = int(os.getenv("CALE_SEED", "1337"))
random.seed(SEED)
np.random.seed(SEED)
if torch is not None:  # pragma: no branch
    try:  # pragma: no cover - guard against broken torch installs
        torch.manual_seed(SEED)
    except Exception:
        pass


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
