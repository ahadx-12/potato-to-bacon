"""Ambiguity metrics based on entropy of interpretation clusters."""
from __future__ import annotations

import math
from typing import Iterable


def normalized_entropy(probabilities: Iterable[float]) -> float:
    """Return entropy normalized by log(K) where K is cluster count."""

    probs = [p for p in probabilities if p > 0]
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    normalized = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in normalized)
    max_entropy = math.log(len(normalized)) if len(normalized) > 1 else 1.0
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)
