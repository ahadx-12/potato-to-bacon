from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .patterns import BYPASS_RE, THRESH_RE, METRIC_WORDS

@dataclass
class BypassHit:
    is_bypass: bool
    trigger: str
    has_threshold: bool
    threshold_value: float | None
    threshold_text: str | None
    metrics: List[str]
    strength: float  # 0..1

def _extract_trigger(s: str) -> str:
    m = BYPASS_RE.search(s)
    return m.group(0).lower() if m else ""

def _extract_threshold(s: str) -> tuple[bool, float | None, str | None]:
    m = THRESH_RE.search(s)
    if not m:
        return False, None, None
    txt = m.group(0)
    val: float | None = None
    if m.group("num"):
        n = float(m.group("num").replace(",", ""))
        unit = (m.group("unit") or "").lower()
        if unit in ("b", "bn", "billion"):
            val = n * 1_000
        elif unit in ("m", "mm", "million"):
            val = n
        elif unit == "%":
            val = n / 100.0
        else:
            # No unit, assume millions if currency present else raw
            val = n
    elif m.group("xnum"):
        val = float(m.group("xnum"))
    return True, val, txt

def _find_metrics(s: str) -> List[str]:
    low = s.lower()
    hits = [w for w in METRIC_WORDS if w in low]
    # de-duplicate by stem-ish
    out: List[str] = []
    for w in hits:
        if all(w not in z and z not in w for z in out):
            out.append(w)
    return out

def detect_bypass(sent: str) -> BypassHit:
    trig = _extract_trigger(sent)
    if not trig:
        return BypassHit(False, "", False, None, None, [], 0.0)
    has_th, th_val, th_txt = _extract_threshold(sent)
    mets = _find_metrics(sent)
    # Heuristic strength
    base = 0.4
    if has_th: base += 0.3
    if mets:   base += min(0.2, 0.05 * len(mets))
    if "notwithstanding" in trig:  # may override rather than carve-out; deem slightly weaker
        base -= 0.1
    return BypassHit(True, trig, has_th, th_val, th_txt, mets, max(0.0, min(1.0, base)))
