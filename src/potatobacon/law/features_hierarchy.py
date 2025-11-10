"""Feature utilities describing the hierarchy of U.S. tax authorities."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

PRECEDENCE_MATRIX = {
    ("statute", "regulation"): 1.0,
    ("statute", "guidance"): 0.8,
    ("statute", "case"): 0.9,
    ("regulation", "guidance"): 0.6,
    ("regulation", "case"): 0.7,
    ("guidance", "case"): 0.5,
}
DEFAULT_SAME_LAYER = 0.5


@dataclass(frozen=True)
class HierarchyContext:
    """Input data for computing hierarchy weightings."""

    layer_a: str
    layer_b: str
    shared_citations: Iterable[str]
    citation_total: int
    newer_date: dt.date | None = None
    older_date: dt.date | None = None


def precedence_weight(layer_a: str, layer_b: str) -> float:
    """Return a precedence multiplier based on the interacting layers."""

    if layer_a == layer_b:
        return DEFAULT_SAME_LAYER
    pair = (layer_a, layer_b)
    reverse = (layer_b, layer_a)
    if pair in PRECEDENCE_MATRIX:
        return PRECEDENCE_MATRIX[pair]
    if reverse in PRECEDENCE_MATRIX:
        return PRECEDENCE_MATRIX[reverse]
    return DEFAULT_SAME_LAYER


def directness_score(shared_citations: Iterable[str], total: int) -> float:
    """Estimate directness from citation overlap."""

    shared_count = len(list(shared_citations))
    if total <= 0:
        return 0.0
    directness = shared_count / total
    return max(0.0, min(1.0, directness))


def temporal_priority(newer: dt.date | None, older: dt.date | None) -> float:
    """Compute a temporal priority multiplier via a logistic mapping."""

    if not newer or not older:
        return 0.5
    delta_days = (newer - older).days
    # Clamp to avoid overflow
    delta_days = max(min(delta_days, 365 * 10), -365 * 10)
    # later enactments should have higher weight (>0.5)
    return 1.0 / (1.0 + pow(2.718281828459045, -delta_days / 365.0))


def hierarchy_term(context: HierarchyContext) -> float:
    """Combine precedence, directness, and temporal priority into a single value."""

    precedence = precedence_weight(context.layer_a, context.layer_b)
    directness = directness_score(context.shared_citations, context.citation_total)
    temporal = temporal_priority(context.newer_date, context.older_date)
    return precedence * max(0.1, directness) * temporal
