from __future__ import annotations
import re
from typing import Tuple
from .patterns import OBLIGATION_POS_WORDS, OBLIGATION_NEG_WORDS

def score_obligation(sent: str) -> Tuple[float, int]:
    """
    Returns (score, polarity)
    score ∈ [0,1], polarity ∈ {+1 (duty), -1 (prohibition), 0 (unknown)}.
    """
    s = sent.lower()
    pos = sum(1 for w in OBLIGATION_POS_WORDS if w in s)
    neg = sum(1 for w in OBLIGATION_NEG_WORDS if w in s)
    raw = pos + neg
    if raw == 0:
        return 0.0, 0
    score = min(1.0, raw / 3.0)
    polarity = -1 if neg > pos else +1
    return score, polarity
