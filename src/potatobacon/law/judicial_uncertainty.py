"""Judicial uncertainty metrics derived from case outcomes."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def judicial_uncertainty(outcomes: Sequence[float], split_circuits: int = 0, total_circuits: int = 0, epsilon: float = 1e-6) -> float:
    """Compute judicial uncertainty as variance/mean adjusted by split circuits."""

    if not outcomes:
        return 0.0
    mean = sum(outcomes) / len(outcomes)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in outcomes) / len(outcomes)
    circuit_term = 0.0
    if total_circuits > 0 and split_circuits > 0:
        circuit_term = epsilon * split_circuits / total_circuits
    return float(max(0.0, variance / (abs(mean) + epsilon) + circuit_term))
