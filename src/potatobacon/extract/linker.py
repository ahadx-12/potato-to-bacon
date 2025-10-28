from __future__ import annotations
from typing import List, Tuple
from .obligation import score_obligation

def link_bypass_to_obligation(sents: List[str], bypass_idx: int, window: int = 3) -> Tuple[int | None, float, int]:
    """
    Returns (ob_idx, link_score, polarity)
    - ob_idx: index of linked obligation or None
    - link_score: 0..1 (higher = stronger link)
    - polarity: +1 duty, -1 prohibition, 0 unknown
    """
    n = len(sents)
    best = (None, 0.0, 0)
    for j in range(max(0, bypass_idx - window), min(n, bypass_idx + window + 1)):
        if j == bypass_idx:
            continue
        score, polarity = score_obligation(sents[j])
        if score <= 0:
            continue
        dist = abs(j - bypass_idx)
        # Distance penalty (closer is better)
        link = score * (1.0 - 0.15 * dist)
        if link > best[1]:
            best = (j, link, polarity)
    return best
