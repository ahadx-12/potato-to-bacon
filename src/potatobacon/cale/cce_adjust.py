from __future__ import annotations
from typing import Tuple

def adjust_cce(base_cce: float, bypass_strength: float, link_score: float, obligation_polarity: int) -> Tuple[float, float]:
    """
    Return (adjusted_cce, delta). Positive polarity = duty; negative = prohibition.
    If bypass undermines the obligation, reduce CCE; if it carves out from a prohibition, also reduce (still undermines enforceability).
    """
    if bypass_strength <= 0 or link_score <= 0:
        return base_cce, 0.0
    # How much do we penalize? weight grows with both strength/link
    w = min(1.0, 0.6 * bypass_strength + 0.6 * link_score)
    delta = - w * abs(base_cce)  # full-scale penalty on magnitude
    new_cce = base_cce + delta
    # If very strong carve-out, allow sign flip to represent negate
    if w > 0.85:
        new_cce = -abs(new_cce)
        delta = new_cce - base_cce
    return new_cce, delta
