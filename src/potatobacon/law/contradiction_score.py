"""Compute contradiction probabilities between legal texts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .features_hierarchy import hierarchy_term, HierarchyContext


def sigmoid(value: float) -> float:
    """Standard logistic function with protection against overflow."""

    if value >= 0:
        exp_neg = pow(2.718281828459045, -value)
        return 1 / (1 + exp_neg)
    exp_pos = pow(2.718281828459045, value)
    return exp_pos / (1 + exp_pos)


@dataclass
class ContradictionFeatures:
    """Container for features feeding the contradiction probability."""

    cosine_similarity: float
    negation_overlap: float
    numeric_conflict: float
    shared_citations: Iterable[str]
    citation_total: int
    layer_a: str
    layer_b: str
    newer_date: object | None = None
    older_date: object | None = None


def contradiction_probability(
    features: ContradictionFeatures,
    alpha: float = 2.0,
    beta: float = 1.5,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> float:
    """Compute the contradiction probability using weighted features.

    Parameters mirror the specification: cosine similarity (alpha), negation overlap (beta),
    numeric infeasibility (gamma), and the hierarchy term (delta).
    """

    hierarchy = hierarchy_term(
        HierarchyContext(
            features.layer_a,
            features.layer_b,
            features.shared_citations,
            features.citation_total,
            features.newer_date,
            features.older_date,
        )
    )
    score = (
        alpha * features.cosine_similarity
        + beta * features.negation_overlap
        + gamma * features.numeric_conflict
        + delta * hierarchy
    )
    return float(max(0.0, min(1.0, sigmoid(score))))
