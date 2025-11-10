"""Aggregate flaw score computations."""
from __future__ import annotations

from typing import Iterable


def flaw_score(
    ambiguity: float,
    contradiction: float,
    judicial: float,
    fragility: float,
    weights: Iterable[float] | None = None,
) -> float:
    """Combine sub-metrics into a single flaw score."""

    components = [ambiguity, contradiction, judicial, fragility]
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    weights_list = list(weights)
    if len(weights_list) != len(components):
        raise ValueError("weights must match component count")
    total = 0.0
    for weight, value in zip(weights_list, components):
        total += max(0.0, weight) * max(0.0, value)
    return total


def policy_flaw_score(flaw: float, impact: float) -> float:
    """Scale the flaw score by normalized impact."""

    norm_flaw = max(0.0, flaw)
    norm_impact = max(0.0, impact)
    return norm_flaw * norm_impact
