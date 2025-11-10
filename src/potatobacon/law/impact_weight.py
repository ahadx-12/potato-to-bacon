"""Impact weighting helper functions."""
from __future__ import annotations

import math


def impact_weight(reach: float, stakes: float) -> float:
    """Compute the impact weight using logarithmic reach and normalized stakes."""

    reach_term = math.log1p(max(0.0, reach))
    stakes_term = max(0.0, min(1.0, stakes))
    return reach_term * stakes_term
