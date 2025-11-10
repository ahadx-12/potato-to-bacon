"""Graph-based fragility metrics for the tax law corpus."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import networkx as nx


def compute_network_scores(edges: Iterable[Tuple[str, str]], damping: float = 0.85) -> Dict[str, float]:
    """Return combined betweenness and PageRank scores for the supplied edges."""

    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    if graph.number_of_nodes() == 0:
        return {}
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    pagerank = nx.pagerank(graph, alpha=damping)
    scores: Dict[str, float] = {}
    for node in graph.nodes:
        scores[node] = 0.5 * betweenness.get(node, 0.0) + 0.5 * pagerank.get(node, 0.0)
    return scores
