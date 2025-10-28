from __future__ import annotations
import re
from typing import List

_SENT_SPLIT = re.compile(r'(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+(?=[A-Z(])')

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Normalize awkward whitespace
    text = re.sub(r'\s+', ' ', text)
    parts = _SENT_SPLIT.split(text)
    # Keep short clauses merged if the split was too aggressive
    merged = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 20 and buf:
            buf += " " + p
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged
