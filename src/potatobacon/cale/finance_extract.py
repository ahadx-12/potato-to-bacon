"""Finance-specific extraction utilities for CALE."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FINANCE_CONFIG_PATH = Path("configs/finance.yml")

try:  # Optional YAML dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


def _load_config() -> Dict[str, object]:
    """Load finance configuration from ``configs/finance.yml`` if present."""

    if yaml is None or not FINANCE_CONFIG_PATH.exists():
        return {}
    try:
        with FINANCE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:  # pragma: no cover - defensive
        return {}
    return data


CFG = _load_config()

DEFAULT_LOGISTIC = [-0.2, 0.7, 0.9, 0.4, 0.3, 0.25]
DEFAULT_SENT_WINDOW = 2
DEFAULT_OVERLAP = 0.35

SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z(])")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

OBL_PATTERNS = [
    re.compile(r"\b(shall|must|is required to|are required to|shall maintain|must maintain|minimize|maintain a)\b", re.I),
    re.compile(r"\b(shall not exceed|must not exceed|shall ensure|must ensure)\b", re.I),
]
PERM_PATTERNS = [
    re.compile(r"\b(may|is permitted to|are permitted to|may incur|may borrow|may issue|may enter into)\b", re.I),
]

BYPASS_TERMS = (
    "unless",
    "except",
    "subject to",
    "provided that",
    "waiver",
    "amendment",
    "notwithstanding",
    "forbearance",
    "cure",
)

FRAGILITY_PATTERNS: List[Tuple[re.Pattern[str], float]] = [
    (re.compile(r"substantial doubt.*going concern", re.I), 3.0),
    (re.compile(r"event of default", re.I), 2.0),
    (re.compile(r"(waiver|forbearance|amendment)", re.I), 1.5),
    (re.compile(r"(leverage ratio|interest coverage|fixed charge coverage|dscr|minim(um)? liquidity)", re.I), 0.8),
    (re.compile(r"accelerat(e|ion)", re.I), 1.0),
]

AUTHORITY_PRIORS = {
    "credit_agreement": 2.0,
    "indenture": 2.0,
    "notes_to_fs": 1.5,
    "liquidity_capital": 1.2,
    "risk_factors": 0.7,
    "mdna": 0.4,
}

if CFG:
    AUTHORITY_PRIORS.update(CFG.get("authority_priors", {}) or {})


@dataclass(slots=True)
class Rule:
    """Structured representation of an extracted obligation or permission."""

    kind: str  # "OBLIGATION" or "PERMISSION"
    verb: str
    object: str
    scope: str
    section: str
    sentence: str
    index: int


def _normalise_tokens(text: str) -> List[str]:
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text)]
    return [_stem(token) for token in tokens if token]


def _stem(token: str) -> str:
    # Tiny Porter-ish stemmer without external deps.
    for suffix in ("ingly", "edly", "ing", "ers", "ies", "ied", "er", "ed", "ly", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    sentences: List[str] = []
    start = 0
    for match in SENT_SPLIT_RE.finditer(text):
        end = match.start()
        chunk = text[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def detect_bypass(sentence: str) -> int:
    lowered = sentence.lower()
    return int(any(term in lowered for term in BYPASS_TERMS))


def section_authority(section_title: str) -> float:
    key = section_title.lower()
    if "credit" in key or "facilities" in key or "debt" in key:
        return float(AUTHORITY_PRIORS.get("credit_agreement", 2.0))
    if "indenture" in key or "note" in key:
        return float(AUTHORITY_PRIORS.get("indenture", 2.0))
    if "notes" in key and "financial" in key:
        return float(AUTHORITY_PRIORS.get("notes_to_fs", 1.5))
    if "liquidity" in key or "capital resources" in key:
        return float(AUTHORITY_PRIORS.get("liquidity_capital", 1.2))
    if "risk" in key and "factor" in key:
        return float(AUTHORITY_PRIORS.get("risk_factors", 0.7))
    if "management" in key or "md&a" in key:
        return float(AUTHORITY_PRIORS.get("mdna", 0.4))
    return 0.2


def fragility_score(text: str) -> float:
    if not text:
        return 0.0
    total = 0.0
    for pattern, weight in FRAGILITY_PATTERNS:
        if pattern.search(text):
            total += weight
    max_score = sum(weight for _, weight in FRAGILITY_PATTERNS)
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, total / max_score))


def extract_rules(text: str, section: str = "", strict: bool = False) -> List[Rule]:
    sentences = sentence_split(text)
    rules: List[Rule] = []
    for idx, sentence in enumerate(sentences):
        lowered = sentence.lower()
        is_obl = any(p.search(sentence) for p in OBL_PATTERNS)
        is_perm = any(p.search(sentence) for p in PERM_PATTERNS)
        if strict:
            is_obl = is_obl or bool(re.search(r"shall\s+(maintain|keep|deliver|provide|repay)", lowered))
            is_perm = is_perm or bool(re.search(r"may\s+(incur|borrow|issue|enter|waive)", lowered))
        if not (is_obl or is_perm):
            continue
        verb = _extract_verb(sentence)
        obj = _extract_object(sentence)
        scope = section or "GENERAL"
        kind = "OBLIGATION" if is_obl else "PERMISSION"
        if is_obl and is_perm:
            rules.append(Rule("OBLIGATION", verb, obj, scope, section, sentence, idx))
            rules.append(Rule("PERMISSION", verb, obj, scope, section, sentence, idx))
        else:
            rules.append(Rule(kind, verb, obj, scope, section, sentence, idx))
    return rules


def _extract_verb(sentence: str) -> str:
    match = re.search(r"\b(shall|must|may|is required to|are required to|is permitted to|are permitted to)\s+(\w+)", sentence, re.I)
    if match:
        return match.group(2).lower()
    match = re.search(r"\b(maintain|comply|keep|meet|incur|borrow|issue|enter)\b", sentence, re.I)
    if match:
        return match.group(1).lower()
    return ""


def _extract_object(sentence: str) -> str:
    match = re.search(r"(shall|must|may|permit(?:ted)? to)\s+(?:not\s+)?(\w+(?:\s+\w+){0,4})", sentence, re.I)
    if match:
        return match.group(2).strip().lower()
    return sentence[:80].strip().lower()


def _section_slices(text: str) -> List[Tuple[str, str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: List[Tuple[str, str]] = []
    current_title = "GENERAL"
    buffer: List[str] = []
    for line in lines:
        upper = line.upper()
        if (line.isupper() and len(line) < 160) or any(
            key in upper
            for key in (
                "LIQUIDITY",
                "MANAGEMENT",
                "RISK",
                "CREDIT",
                "DEBT",
                "FACILITIES",
                "NOTES",
                "GOING CONCERN",
            )
        ):
            if buffer:
                sections.append((current_title, " ".join(buffer)))
                buffer = []
            current_title = line
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_title, " ".join(buffer)))
    if not sections:
        return [("GENERAL", text)]
    return sections


def _jaccard(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union


def pair_conflict(
    obligation: Rule,
    permission: Rule,
    *,
    sentence_window: int = DEFAULT_SENT_WINDOW,
    overlap_threshold: float = DEFAULT_OVERLAP,
) -> Tuple[float, float]:
    if obligation.section != permission.section:
        if abs(obligation.index - permission.index) > sentence_window:
            return 0.0, 0.0
    tokens_o = _normalise_tokens(obligation.sentence)
    tokens_p = _normalise_tokens(permission.sentence)
    overlap = _jaccard(tokens_o, tokens_p)
    if overlap < overlap_threshold:
        return 0.0, overlap
    conflict_strength = min(1.0, overlap + 0.2)
    return conflict_strength, overlap


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_cce(
    *,
    conflict: float,
    authority: float,
    fragility: float,
    bypass: float,
    delta_prev: float = 0.0,
    lambda_bypass: float = 0.5,
    logistic_weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    authority_norm = _sigmoid(authority)
    cce_prod = max(0.0, min(1.0, conflict * authority_norm * fragility * (1.0 + lambda_bypass * bypass)))
    weights = list(logistic_weights or DEFAULT_LOGISTIC)
    if len(weights) < 6:
        weights = DEFAULT_LOGISTIC
    z = (
        weights[0]
        + weights[1] * conflict
        + weights[2] * authority_norm
        + weights[3] * fragility
        + weights[4] * float(bypass or 0.0)
        + weights[5] * float(delta_prev)
    )
    cce_log = _sigmoid(z)
    return {
        "CCE_prod": cce_prod,
        "CCE_logistic": cce_log,
        "Ab_norm": authority_norm,
    }


def analyse_finance_sections(
    doc_text: str,
    *,
    prior_text: Optional[str] = None,
    section_hint: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, object]:
    config = CFG
    lambda_bypass = float(config.get("lambda_bypass", 0.5))
    overlap_threshold = float(config.get("overlap_threshold", DEFAULT_OVERLAP))
    sentence_window = int(config.get("scope_sentence_window", DEFAULT_SENT_WINDOW))
    logistic_weights: Optional[Sequence[float]] = None
    if isinstance(config.get("logistic_weights"), Sequence):
        logistic_weights = config.get("logistic_weights")  # type: ignore[assignment]

    sections = _section_slices(doc_text)
    evidence: List[Dict[str, object]] = []
    best_conflict = 0.0
    best_overlap = 0.0
    best_authority = 0.0
    best_fragility = 0.0
    bypass_flag = 0

    for section_title, body in sections:
        rules = extract_rules(body, section=section_title, strict=strict)
        obligations = [r for r in rules if r.kind == "OBLIGATION"]
        permissions = [r for r in rules if r.kind == "PERMISSION"]
        if not obligations or not permissions:
            continue
        authority = section_authority(section_title)
        frag = fragility_score(body)
        for obl in obligations:
            for perm in permissions:
                conflict_strength, overlap = pair_conflict(
                    obl,
                    perm,
                    sentence_window=sentence_window,
                    overlap_threshold=overlap_threshold,
                )
                if conflict_strength <= 0.0:
                    continue
                bypass_here = max(detect_bypass(perm.sentence), detect_bypass(obl.sentence))
                bypass_flag = max(bypass_flag, bypass_here)
                if conflict_strength > best_conflict:
                    best_conflict = conflict_strength
                    best_overlap = overlap
                    best_authority = authority
                    best_fragility = frag
                evidence.append(
                    {
                        "section": section_title,
                        "obligation": obl.sentence,
                        "permission": perm.sentence,
                        "bypass": bypass_here,
                        "authority": authority,
                        "fragility": frag,
                        "conflict": conflict_strength,
                        "overlap": overlap,
                    }
                )
    if not evidence:
        frag_full = fragility_score(doc_text)
        base_features = {
            "conflict_intensity": 0.0,
            "semantic_overlap": 0.0,
            "authority_balance": _sigmoid(best_authority),
            "bypass": bypass_flag,
            "fragility": frag_full,
            "CCE_prod": 0.0,
            "CCE_logistic": 0.0,
            "evidence": [],
            "delta_cce": 0.0,
        }
        return base_features

    prev_cce = 0.0
    if prior_text:
        prior = analyse_finance_sections(prior_text, strict=strict)
        prev_cce = float(prior.get("CCE_prod", 0.0))

    base_cce = compute_cce(
        conflict=best_conflict,
        authority=best_authority,
        fragility=best_fragility,
        bypass=bypass_flag,
        delta_prev=0.0,
        lambda_bypass=lambda_bypass,
        logistic_weights=logistic_weights,
    )
    delta_cce = base_cce["CCE_prod"] - prev_cce
    final_cce = compute_cce(
        conflict=best_conflict,
        authority=best_authority,
        fragility=best_fragility,
        bypass=bypass_flag,
        delta_prev=delta_cce,
        lambda_bypass=lambda_bypass,
        logistic_weights=logistic_weights,
    )

    result: Dict[str, object] = {
        "conflict_intensity": best_conflict,
        "semantic_overlap": best_overlap,
        "authority_balance": final_cce["Ab_norm"],
        "fragility": best_fragility,
        "bypass": bypass_flag,
        "CCE_prod": base_cce["CCE_prod"],
        "CCE_logistic": final_cce["CCE_logistic"],
        "delta_cce": delta_cce,
        "evidence": sorted(
            evidence,
            key=lambda item: item.get("conflict", 0.0),
            reverse=True,
        )[:10],
    }

    return result


__all__ = [
    "Rule",
    "extract_rules",
    "detect_bypass",
    "section_authority",
    "fragility_score",
    "pair_conflict",
    "compute_cce",
    "analyse_finance_sections",
]
