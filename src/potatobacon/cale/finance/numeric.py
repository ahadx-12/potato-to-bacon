"""Numeric covenant extraction utilities for CALE finance workflows."""

from __future__ import annotations

import importlib
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CFG_PATH = Path(__file__).resolve().parents[4] / "configs" / "finance.yml"

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


DEFAULT_METRIC_ALIASES: Dict[str, Tuple[str, Optional[str]]] = {
    "total leverage": ("LEVERAGE", "TOTAL"),
    "net leverage": ("LEVERAGE", "NET"),
    "secured leverage": ("LEVERAGE", "SECURED"),
    "first lien leverage": ("LEVERAGE", "FIRST_LIEN"),
    "first-lien leverage": ("LEVERAGE", "FIRST_LIEN"),
    "first lien net leverage": ("LEVERAGE", "FIRST_LIEN"),
    "first-lien net leverage": ("LEVERAGE", "FIRST_LIEN"),
    "senior secured leverage": ("LEVERAGE", "SENIOR_SECURED"),
    "senior secured net leverage": ("LEVERAGE", "SENIOR_SECURED"),
    "total leverage ratio": ("LEVERAGE", "TOTAL"),
    "interest coverage": ("COVERAGE", "INTEREST"),
    "ebitda to interest": ("COVERAGE", "INTEREST"),
    "ebit interest coverage": ("COVERAGE", "INTEREST"),
    "fixed charge coverage": ("COVERAGE", "FIXED_CHARGE"),
    "debt service coverage": ("COVERAGE", "DSCR"),
    "debt service coverage ratio": ("COVERAGE", "DSCR"),
    "dscr": ("COVERAGE", "DSCR"),
    "minimum liquidity": ("LIQUIDITY_MIN", None),
    "liquidity": ("LIQUIDITY_MIN", None),
    "cash balance": ("LIQUIDITY_MIN", None),
    "cash and cash equivalents": ("LIQUIDITY_MIN", None),
    "capital expenditures": ("CAPEX_MAX", None),
    "capital expenditure": ("CAPEX_MAX", None),
    "capex": ("CAPEX_MAX", None),
    "investments": ("INVESTMENTS_MAX", None),
    "restricted payments": ("RESTRICTED_PAYMENTS", None),
    "restricted payment": ("RESTRICTED_PAYMENTS", None),
    "dividends": ("RESTRICTED_PAYMENTS", "DIVIDENDS"),
    "tangible net worth": ("TANGIBLE_NET_WORTH", None),
    "net working capital": ("NET_WORKING_CAPITAL", None),
    "current ratio": ("CURRENT_RATIO", None),
    "quick ratio": ("QUICK_RATIO", None),
    "asset coverage": ("ASSET_COVERAGE", None),
    "loan to value": ("LTV", None),
    "loan-to-value": ("LTV", None),
    "ltv": ("LTV", None),
}

DEFAULT_QUALIFIER_TERMS: Dict[str, Any] = {
    "pro_forma": ["pro forma"],
    "ttm": ["trailing twelve months", "ttm"],
    "consolidated": ["consolidated"],
    "net": ["net"],
    "secured": {
        "SENIOR": ["senior secured"],
        "FIRST_LIEN": ["first lien", "first-lien"],
        "SECURED": ["secured"],
    },
}


def _load_config() -> Dict[str, Any]:
    spec = importlib.util.find_spec("yaml")
    if spec is None or not CFG_PATH.exists():
        return {}
    yaml = importlib.import_module("yaml")
    with CFG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


CFG = _load_config()

METRIC_ALIASES: Dict[str, Tuple[str, Optional[str]]] = dict(DEFAULT_METRIC_ALIASES)
if isinstance(CFG.get("metric_aliases"), dict):
    for key, value in CFG["metric_aliases"].items():
        if isinstance(value, (list, tuple)) and value:
            metric = str(value[0]).upper()
            subtype = (
                str(value[1]).upper()
                if len(value) >= 2 and value[1] not in (None, "")
                else None
            )
            METRIC_ALIASES[str(key).lower()] = (metric, subtype)

QUALIFIER_TERMS: Dict[str, Any] = dict(DEFAULT_QUALIFIER_TERMS)
if isinstance(CFG.get("qualifier_terms"), dict):
    for key, value in CFG["qualifier_terms"].items():
        if key == "secured" and isinstance(value, dict):
            merged = QUALIFIER_TERMS.get("secured", {}).copy()
            for sub_key, items in value.items():
                if isinstance(items, (list, tuple)):
                    merged[str(sub_key).upper()] = [str(item).lower() for item in items]
            QUALIFIER_TERMS["secured"] = merged
        elif isinstance(value, (list, tuple)):
            QUALIFIER_TERMS[str(key)] = [str(item).lower() for item in value]


# ---------------------------------------------------------------------------
# Regex bank
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z0-9\(])")
PAREN_NUM_RE = re.compile(r"\((\d{1,3}(?:\.\d{1,4})?)\)")
RATIO_RE = re.compile(r"(\d{1,2}(?:\.\d{1,3})?)\s*(?:x|times)\b", re.I)
RATIO_TO_ONE_RE = re.compile(r"(\d{1,2}(?:\.\d{1,3})?)\s*(?:to|:)\s*1(?:\.0+)?\b", re.I)
PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,4})?)\s*%", re.I)
BPS_RE = re.compile(r"(\d{1,4})\s*bps", re.I)
CURRENCY_RE = re.compile(
    r"\$?\s*(\d{1,3}(?:[\s\u00A0,]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*(million|billion|mm|bn|m|b))?",
    re.I,
)
BETWEEN_RE = re.compile(r"(?i)\bbetween\s+(.+?)\s+and\s+(.+?)(?=(?:[,;]|\.(?:\s|$)|$))")
COMBINATOR_RE = re.compile(r"(?i)\b(?:the\s+)?(greater|lesser)\s+of\b")
COMBINATOR_SPLIT_RE = re.compile(r"(?:(?:;|,)\s*|\s+and\s+)")
OP_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = (
    (
        re.compile(
            r"(?i)\b(?:(?:shall|will|may)\s+not\s+exceed|"
            r"not\s+to\s+exceed|"
            r"(?:shall|will)\s+not\s+be\s+greater\s+than|"
            r"no\s+more\s+than|no\s+greater\s+than|"
            r"maximum(?:\s+of)?|capped\s+at|cap\s+at|max\s*\b)"
        ),
        "<=",
    ),
    (
        re.compile(
            r"(?i)\b(?:(?:shall|will)\s+not\s+be\s+less\s+than|"
            r"not\s+be\s+less\s+than|not\s+less\s+than|"
            r"at\s+least|minimum(?:\s+of)?|no\s+less\s+than)"
        ),
        ">=",
    ),
)
BETWEEN_OPERATOR_RE = re.compile(r"(?i)\bbetween\b")
GREATER_LESSER_RE = re.compile(r"(?i)\b(?:the\s+)?(greater|lesser)\s+of\b")
BASIS_RE = re.compile(r"(?i)\bof\s+([A-Za-z\s]+?)(?:[,.;]|$)")


@dataclass
class ParsedNumber:
    value: float
    unit: str
    span: Tuple[int, int]
    text: str
    basis: Optional[str] = None


@dataclass
class ParsedCovenant:
    metric: str
    subtype: Optional[str]
    op: str
    unit: Optional[str]
    value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    combiner: Optional[str]
    legs: Optional[List[Dict[str, Any]]]
    qualifiers: Dict[str, Any]
    raw_sentence: str
    spans: List[Tuple[int, int]]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "subtype": self.subtype,
            "op": self.op,
            "unit": self.unit,
            "value": self.value,
            "min": self.min_value,
            "max": self.max_value,
            "combiner": self.combiner,
            "legs": self.legs,
            "qualifiers": self.qualifiers,
            "raw": {"sentence": self.raw_sentence, "spans": self.spans, "text": self.raw_sentence},
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sentence_iter(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _resolve_metric(sentence: str) -> Optional[Tuple[str, Optional[str]]]:
    lowered = sentence.lower()
    matches: List[Tuple[int, Tuple[str, Optional[str]]]] = []
    for phrase, mapping in METRIC_ALIASES.items():
        if phrase in lowered:
            matches.append((len(phrase), mapping))
    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    # Heuristic fallbacks
    if "leverage" in lowered:
        subtype: Optional[str] = "TOTAL"
        if "net leverage" in lowered:
            subtype = "NET"
        if "first lien" in lowered or "first-lien" in lowered:
            subtype = "FIRST_LIEN"
        elif "senior secured" in lowered:
            subtype = "SENIOR_SECURED"
        elif "secured" in lowered:
            subtype = "SECURED"
        return "LEVERAGE", subtype
    if "coverage" in lowered or "dscr" in lowered:
        if "fixed charge" in lowered:
            return "COVERAGE", "FIXED_CHARGE"
        if "debt service" in lowered or "dscr" in lowered:
            return "COVERAGE", "DSCR"
        return "COVERAGE", "INTEREST"
    if "liquidity" in lowered or "cash balance" in lowered:
        return "LIQUIDITY_MIN", None
    if "capital expenditure" in lowered or "capex" in lowered:
        return "CAPEX_MAX", None
    if "investment" in lowered:
        return "INVESTMENTS_MAX", None
    if "restricted payment" in lowered or "dividend" in lowered:
        return "RESTRICTED_PAYMENTS", None
    if "tangible net worth" in lowered:
        return "TANGIBLE_NET_WORTH", None
    if "net working capital" in lowered:
        return "NET_WORKING_CAPITAL", None
    if "current ratio" in lowered:
        return "CURRENT_RATIO", None
    if "quick ratio" in lowered:
        return "QUICK_RATIO", None
    if "asset coverage" in lowered:
        return "ASSET_COVERAGE", None
    if "loan to value" in lowered or "loan-to-value" in lowered or "ltv" in lowered:
        return "LTV", None
    return None


def _normalize_basis(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"[^A-Za-z\s]", "", text).strip()
    if not cleaned:
        return None
    return re.sub(r"\s+", " ", cleaned).upper()


def _parse_ratio(sentence: str) -> Optional[ParsedNumber]:
    match = RATIO_RE.search(sentence)
    if match:
        value = float(match.group(1))
        return ParsedNumber(value=value, unit="x", span=match.span(), text=match.group(0))
    match = RATIO_TO_ONE_RE.search(sentence)
    if match:
        value = float(match.group(1))
        return ParsedNumber(value=value, unit="x", span=match.span(), text=match.group(0))
    # Parenthetical override combined with "to one"
    paren = PAREN_NUM_RE.search(sentence)
    if paren and "to one" in sentence.lower():
        try:
            value = float(paren.group(1))
        except ValueError:
            return None
        return ParsedNumber(value=value, unit="x", span=paren.span(), text=paren.group(0))
    return None


def _scale_currency(value: float, scale: Optional[str]) -> float:
    if not scale:
        return value
    scale = scale.lower()
    if scale in {"million", "mm", "m"}:
        return value * 1_000_000
    if scale in {"billion", "bn", "b"}:
        return value * 1_000_000_000
    return value


def _parse_currency(sentence: str) -> Optional[ParsedNumber]:
    for match in CURRENCY_RE.finditer(sentence):
        raw_value = match.group(1)
        if raw_value is None:
            continue
        normalized = re.sub(r"[\s\u00A0]", "", raw_value.replace(",", ""))
        try:
            value = float(normalized)
        except ValueError:
            continue
        value = _scale_currency(value, match.group(2))
        return ParsedNumber(value=value, unit="USD", span=match.span(), text=match.group(0))
    return None


def _parse_percent(sentence: str) -> Optional[ParsedNumber]:
    match = PERCENT_RE.search(sentence)
    if match:
        value = float(match.group(1)) / 100.0
        return ParsedNumber(value=value, unit="PCT", span=match.span(), text=match.group(0))
    match = BPS_RE.search(sentence)
    if match:
        value = float(match.group(1)) / 10_000.0
        return ParsedNumber(value=value, unit="PCT", span=match.span(), text=match.group(0))
    return None


def _detect_operator(sentence: str) -> Optional[str]:
    if BETWEEN_OPERATOR_RE.search(sentence):
        return "BETWEEN"
    if GREATER_LESSER_RE.search(sentence):
        match = GREATER_LESSER_RE.search(sentence)
        if match:
            kind = match.group(1).lower()
            return "GREATER_OF" if kind.startswith("greater") else "LESSER_OF"
    for pattern, op in OP_PATTERNS:
        if pattern.search(sentence):
            return op
    return None


def _extract_qualifiers(sentence: str, metric: Optional[str]) -> Dict[str, Any]:
    lowered = sentence.lower()
    qualifiers = {
        "pro_forma": False,
        "ttm": False,
        "consolidated": False,
        "net": False,
        "secured": None,
    }
    for key, terms in QUALIFIER_TERMS.items():
        if key == "secured":
            secured_terms = QUALIFIER_TERMS.get("secured", {})
            for secured_key, variants in secured_terms.items():
                for variant in variants:
                    if variant.lower() in lowered:
                        qualifiers["secured"] = secured_key
                        break
                if qualifiers["secured"]:
                    break
        else:
            if isinstance(terms, Sequence):
                for term in terms:
                    if term.lower() in lowered:
                        qualifiers[key] = True
                        break
    if metric == "LEVERAGE" and "net leverage" in lowered:
        qualifiers["net"] = True
    return qualifiers


def _parse_between(sentence: str) -> Optional[Tuple[ParsedNumber, ParsedNumber]]:
    match = BETWEEN_RE.search(sentence)
    if not match:
        return None
    left = match.group(1)
    right = match.group(2)
    left_num = _parse_number_token(left)
    right_num = _parse_number_token(right)
    if left_num and right_num and left_num.unit == right_num.unit:
        return left_num, right_num
    return None


def _parse_number_token(token: str) -> Optional[ParsedNumber]:
    token = token.strip()
    if not token:
        return None
    ratio = _parse_ratio(token)
    if ratio:
        return ratio
    percent = _parse_percent(token)
    if percent:
        return percent
    currency = _parse_currency(token)
    if currency:
        return currency
    return None


def _parse_simple_value(sentence: str, metric: Optional[str]) -> Optional[ParsedNumber]:
    preferred_units: List[str] = []
    lowered = sentence.lower()
    has_percent = "%" in sentence or "percent" in lowered
    if metric in {"LEVERAGE", "COVERAGE", "CURRENT_RATIO", "QUICK_RATIO", "ASSET_COVERAGE"}:
        preferred_units = ["x"]
    elif metric in {"LTV"}:
        preferred_units = ["PCT", "x"]
    elif metric in {"LIQUIDITY_MIN", "CAPEX_MAX", "INVESTMENTS_MAX", "RESTRICTED_PAYMENTS", "TANGIBLE_NET_WORTH", "NET_WORKING_CAPITAL"}:
        preferred_units = ["USD", "PCT"]
    else:
        preferred_units = ["x", "USD", "PCT"]

    candidates: List[ParsedNumber] = []
    ratio = _parse_ratio(sentence)
    if ratio:
        candidates.append(ratio)
    percent = _parse_percent(sentence)
    if percent:
        candidates.append(percent)
    currency = _parse_currency(sentence)
    if currency:
        candidates.append(currency)

    if not candidates:
        return None

    # Choose candidate based on preferred units order
    if has_percent and "PCT" in preferred_units:
        preferred_units = [unit for unit in preferred_units if unit != "PCT"]
        preferred_units.insert(0, "PCT")

    for unit in preferred_units:
        for candidate in candidates:
            if candidate.unit == unit:
                return candidate
    return candidates[0]


def _parse_combinator(sentence: str) -> Optional[Tuple[str, List[Dict[str, Any]], List[Tuple[int, int]]]]:
    match = COMBINATOR_RE.search(sentence)
    if not match:
        return None
    kind = match.group(1).lower()
    remainder = sentence[match.end():]
    cleaned = re.sub(r"(?i)\b(?:i{1,3}|iv|v|vi|1|2|3)\)\s*", "", remainder)
    parts = [part.strip(" ,;:") for part in COMBINATOR_SPLIT_RE.split(cleaned) if part.strip()]
    legs: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for part in parts:
        number = _parse_number_token(part)
        if not number:
            continue
        basis_match = BASIS_RE.search(part)
        basis = _normalize_basis(basis_match.group(1)) if basis_match else None
        leg: Dict[str, Any] = {"value": number.value, "unit": number.unit}
        if basis:
            leg["basis"] = basis
        legs.append(leg)
        # approximate span by searching original sentence
        sub_match = re.search(re.escape(part), sentence)
        if sub_match:
            spans.append(sub_match.span())
        else:
            spans.append(number.span)
    if len(legs) >= 2:
        op = "GREATER_OF" if kind.startswith("greater") else "LESSER_OF"
        return op, legs, spans
    return None


def _coalesce_results(parsed: ParsedCovenant) -> Dict[str, Any]:
    return parsed.to_dict()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_numeric_covenants(text: str) -> List[Dict[str, Any]]:
    """Extract normalized numeric covenants from the given text."""

    results: List[Dict[str, Any]] = []
    for sentence in _sentence_iter(text):
        metric_info = _resolve_metric(sentence)
        if not metric_info:
            continue
        metric, subtype = metric_info
        qualifiers = _extract_qualifiers(sentence, metric)

        op = _detect_operator(sentence)
        spans: List[Tuple[int, int]] = []
        value: Optional[float] = None
        unit: Optional[str] = None
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        combiner: Optional[str] = None
        legs: Optional[List[Dict[str, Any]]] = None

        if op == "BETWEEN":
            between = _parse_between(sentence)
            if not between:
                continue
            left, right = between
            min_value, max_value = left.value, right.value
            unit = left.unit
            spans.extend([left.span, right.span])
        elif op in {"GREATER_OF", "LESSER_OF"}:
            comb_data = _parse_combinator(sentence)
            if not comb_data:
                continue
            combiner, legs, comb_spans = comb_data
            op = combiner
            spans.extend(comb_spans)
        else:
            number = _parse_simple_value(sentence, metric)
            if not number:
                continue
            value = number.value
            unit = number.unit
            spans.append(number.span)
            if op is None:
                # infer direction from metric keywords when operator missing but cues exist
                lowered = sentence.lower()
                if metric == "LIQUIDITY_MIN" and (
                    "minimum" in lowered or "at least" in lowered or "not less than" in lowered
                ):
                    op = ">="
                elif metric in {"CAPEX_MAX", "INVESTMENTS_MAX", "RESTRICTED_PAYMENTS"} and (
                    "not exceed" in lowered
                    or "no more than" in lowered
                    or "maximum" in lowered
                    or "capped" in lowered
                    or "cap" in lowered
                ):
                    op = "<="
                elif "between" in lowered:
                    op = "BETWEEN"
                else:
                    op = "=="

        confidence = 0.0
        if op:
            confidence += 0.35
        if metric:
            confidence += 0.35
        if value is not None or min_value is not None or combiner is not None or legs:
            confidence += 0.2
        if qualifiers.get("consolidated") or qualifiers.get("pro_forma"):
            confidence += 0.05
        confidence = float(min(1.0, max(0.0, confidence)))

        if confidence < 0.5:
            continue

        parsed = ParsedCovenant(
            metric=metric,
            subtype=subtype,
            op=op or "==",
            unit=unit,
            value=value,
            min_value=min_value,
            max_value=max_value,
            combiner=combiner,
            legs=legs,
            qualifiers=qualifiers,
            raw_sentence=sentence,
            spans=spans,
            confidence=confidence,
        )
        results.append(_coalesce_results(parsed))
    return results


__all__ = ["extract_numeric_covenants"]
