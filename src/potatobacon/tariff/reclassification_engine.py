from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from potatobacon.tariff.rate_store import RateEntry, get_rate_store
from potatobacon.tariff.hts_ingest.usitc_parser import duty_rate_as_float

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_DIGIT_RE = re.compile(r"\D")

MATERIAL_CHAPTER_HINTS: Dict[str, tuple[str, ...]] = {
    "steel": ("72", "73"),
    "stainless": ("72", "73"),
    "iron": ("72", "73"),
    "aluminum": ("76",),
    "aluminium": ("76",),
    "plastic": ("39",),
    "polymer": ("39",),
    "rubber": ("40",),
    "textile": tuple(f"{idx:02d}" for idx in range(50, 64)),
    "cotton": tuple(f"{idx:02d}" for idx in range(50, 64)),
    "wool": tuple(f"{idx:02d}" for idx in range(50, 64)),
    "electronics": ("84", "85", "90"),
    "battery": ("85",),
    "machinery": ("84",),
    "vehicle": ("87",),
    "furniture": ("94",),
    "food": ("02", "03", "04", "09", "15", "16", "20"),
    "chemical": ("28", "29", "32", "34", "38"),
}

RISK_ORDER = {"low_risk": 0, "medium_risk": 1, "high_risk": 2}


@dataclass(frozen=True)
class HTSCandidate:
    hts_code: str
    description: str
    duty_rate: float
    confidence: float
    score: float


def _normalize_hts(code: str | None) -> str:
    if not code:
        return ""
    return _DIGIT_RE.sub("", code)


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_WORD_RE.findall(text.lower()))


def _token_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    inter = len(left_set & right_set)
    return inter / max(1, len(left_set))


def _risk_level(current_digits: str, candidate_digits: str) -> tuple[str, str]:
    if not current_digits:
        return "high_risk", "No declared baseline HTS classification available."
    if current_digits[:4] == candidate_digits[:4]:
        return "low_risk", "Within same heading; typically lower challenge risk."
    if current_digits[:2] == candidate_digits[:2]:
        return "medium_risk", "Different heading in same chapter; confirm with a ruling request."
    return "high_risk", "Cross-chapter move; a CBP binding ruling is strongly recommended."


def _difficulty_for_risk(risk_level: str) -> str:
    if risk_level == "low_risk":
        return "low"
    if risk_level == "medium_risk":
        return "medium"
    return "high"


def _chapter_hints_from_material(material_text: str) -> set[str]:
    lowered = material_text.lower()
    chapters: set[str] = set()
    for token, chapter_codes in MATERIAL_CHAPTER_HINTS.items():
        if token in lowered:
            chapters.update(chapter_codes)
    return chapters


def _rate_entries() -> list[RateEntry]:
    store = get_rate_store()
    return store.iter_entries()


def classify_hts_from_description(
    *,
    description: str,
    material: str = "",
    intended_use: str = "",
    top_k: int = 3,
) -> list[HTSCandidate]:
    """Return top HTS candidates using deterministic token overlap."""

    query_tokens = _tokens(" ".join([description, material, intended_use]))
    material_tokens = _tokens(material)
    chapter_hints = _chapter_hints_from_material(material)

    scored: list[HTSCandidate] = []
    for entry in _rate_entries():
        duty_rate = duty_rate_as_float(entry.general_rate)
        if duty_rate is None:
            continue
        digits = _normalize_hts(entry.hts_code)
        if len(digits) < 4:
            continue
        if chapter_hints and digits[:2] not in chapter_hints:
            continue

        desc_tokens = _tokens(entry.description)
        if not desc_tokens:
            continue

        overlap = _token_overlap(query_tokens, desc_tokens)
        if overlap <= 0:
            continue
        material_overlap = _token_overlap(material_tokens, desc_tokens)
        chapter_boost = 0.12 if chapter_hints and digits[:2] in chapter_hints else 0.0
        score = overlap + material_overlap * 1.5 + chapter_boost
        confidence = min(1.0, max(0.0, score))

        scored.append(
            HTSCandidate(
                hts_code=entry.hts_code,
                description=entry.description,
                duty_rate=float(duty_rate),
                confidence=confidence,
                score=score,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.duty_rate, item.hts_code))
    return scored[: max(1, top_k)]


def build_auto_classification_payload(
    *,
    description: str,
    material: str = "",
    intended_use: str = "",
    declared_hts: str | None = None,
    confidence_threshold: float = 0.62,
) -> dict[str, Any]:
    candidates = classify_hts_from_description(
        description=description,
        material=material,
        intended_use=intended_use,
        top_k=3,
    )
    top = candidates[0] if candidates else None
    declared_digits = _normalize_hts(declared_hts)
    top_digits = _normalize_hts(top.hts_code if top else "")

    mismatch_flag = bool(declared_digits and top_digits and not top_digits.startswith(declared_digits[:4]))
    classification_mode = "provided"
    source = "declared"
    selected_hts = declared_hts
    review_reason = None
    confidence = 1.0

    if not declared_hts:
        if top and top.confidence >= confidence_threshold:
            classification_mode = "auto"
            source = "auto_classified"
            selected_hts = top.hts_code
            confidence = top.confidence
        else:
            classification_mode = "manual_review"
            source = "unresolved"
            selected_hts = top.hts_code if top else None
            confidence = top.confidence if top else 0.0
            review_reason = "Top classification confidence is below auto-apply threshold."
    else:
        confidence = top.confidence if top else 0.0
        if mismatch_flag:
            review_reason = "Declared HTS appears inconsistent with description/material signals."

    return {
        "classification": classification_mode,
        "hts_source": source,
        "selected_hts_code": selected_hts,
        "declared_hts_code": declared_hts,
        "confidence": confidence,
        "needs_manual_review": classification_mode == "manual_review" or mismatch_flag,
        "review_reason": review_reason,
        "mismatch_flag": mismatch_flag,
        "alternatives": [
            {
                "hts_code": item.hts_code,
                "description": item.description,
                "duty_rate": item.duty_rate,
                "confidence": round(item.confidence, 4),
            }
            for item in candidates
        ],
    }


def _materials_near_boundary(material_breakdown: Sequence[Mapping[str, Any]]) -> tuple[str, str] | None:
    if len(material_breakdown) < 2:
        return None
    normalized: list[tuple[str, float]] = []
    for item in material_breakdown:
        material = str(item.get("material") or "").lower()
        weight = item.get("percent_by_weight")
        if not material or weight is None:
            continue
        try:
            normalized.append((material, float(weight)))
        except (TypeError, ValueError):
            continue
    if len(normalized) < 2:
        return None
    normalized.sort(key=lambda pair: pair[1], reverse=True)
    lead, second = normalized[0], normalized[1]
    if abs(lead[1] - second[1]) <= 8.0:
        return lead[0], second[0]
    return None


def build_reclassification_candidates(
    *,
    current_hts: str | None,
    baseline_rate: float | None,
    description: str,
    material: str = "",
    annual_volume: int | None = None,
    declared_value_per_unit: float | None = None,
    material_breakdown: Sequence[Mapping[str, Any]] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search same-chapter and cross-chapter lower-duty alternatives."""

    current_digits = _normalize_hts(current_hts)
    if len(current_digits) < 4:
        return []
    if baseline_rate is None:
        return []

    query_tokens = _tokens(f"{description} {material}")
    chapters = {current_digits[:2]}
    chapters.update(_chapter_hints_from_material(material))

    if material_breakdown:
        boundary = _materials_near_boundary(material_breakdown)
        if boundary:
            chapters.update(_chapter_hints_from_material(" ".join(boundary)))

    candidates: list[dict[str, Any]] = []
    for entry in _rate_entries():
        rate = duty_rate_as_float(entry.general_rate)
        if rate is None:
            continue
        rate = float(rate)
        if rate >= baseline_rate:
            continue

        digits = _normalize_hts(entry.hts_code)
        if len(digits) < 4:
            continue
        if digits[:2] not in chapters:
            continue

        overlap = _token_overlap(query_tokens, _tokens(entry.description))
        if overlap < 0.18:
            continue

        risk_level, risk_rationale = _risk_level(current_digits, digits)
        savings_rate = baseline_rate - rate
        savings_per_unit = savings_rate / 100.0 * float(declared_value_per_unit or 0.0)
        annual_savings = None
        if annual_volume is not None and declared_value_per_unit is not None:
            annual_savings = savings_per_unit * annual_volume

        candidates.append(
            {
                "strategy_type": "reclassification",
                "from_hts": current_hts,
                "to_hts": entry.hts_code,
                "candidate_description": entry.description,
                "optimized_duty_rate": rate,
                "savings_per_unit_rate": savings_rate,
                "savings_per_unit_value": savings_per_unit,
                "annual_savings_value": annual_savings,
                "plausibility_score": round(overlap, 4),
                "confidence_level": round(min(1.0, overlap + 0.2), 4),
                "risk_level": risk_level,
                "risk_rationale": risk_rationale,
                "required_actions": [
                    "Validate heading text against product function and materials.",
                    "Prepare broker memo comparing current vs proposed classification.",
                ],
                "documentation_required": [
                    "Detailed BOM with dominant material evidence",
                    "Product datasheet and photos",
                ],
                "implementation_difficulty": _difficulty_for_risk(risk_level),
            }
        )

    candidates.sort(
        key=lambda item: (
            -(item.get("annual_savings_value") or item["savings_per_unit_value"]),
            -item["plausibility_score"],
            RISK_ORDER[item["risk_level"]],
            item["to_hts"],
        )
    )
    return candidates[: max(1, top_k)]


def build_advisory_strategies(
    *,
    origin_country: str | None,
    bom_items: Sequence[Mapping[str, Any]],
    baseline_rate: float | None,
    declared_value_per_unit: float | None,
    annual_volume: int | None,
    material_breakdown: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    origin = (origin_country or "").upper()

    if origin in {"CN", "VN", "TH", "BD"} and baseline_rate is not None and declared_value_per_unit is not None:
        shifted_rate = max(0.0, baseline_rate - min(2.0, baseline_rate))
        savings_rate = max(0.0, baseline_rate - shifted_rate)
        savings_value = savings_rate / 100.0 * declared_value_per_unit
        annual_savings = savings_value * annual_volume if annual_volume is not None else None
        strategies.append(
            {
                "strategy_type": "origin_shift",
                "human_summary": "Evaluate sourcing shift to an FTA partner (e.g., MX/CA/KR) for lower duty treatment.",
                "optimized_duty_rate": shifted_rate,
                "savings_per_unit_rate": savings_rate,
                "savings_per_unit_value": savings_value,
                "annual_savings_value": annual_savings,
                "confidence_level": 0.58,
                "risk_level": "medium_risk",
                "risk_rationale": "Requires rules-of-origin qualification and supply-chain feasibility validation.",
                "required_actions": [
                    "Run country-of-origin qualification analysis for target FTA partner.",
                    "Model landed-cost and production feasibility for the shift.",
                ],
                "documentation_required": [
                    "FTA certificate of origin package",
                    "BOM and transformation records",
                ],
                "implementation_difficulty": "medium",
            }
        )

    if origin in {"CN", "VN", "BD"}:
        strategies.append(
            {
                "strategy_type": "first_sale_valuation",
                "human_summary": (
                    "First sale valuation may reduce dutiable value if purchases route through a middleman."
                ),
                "optimized_duty_rate": baseline_rate,
                "savings_per_unit_rate": 0.0,
                "savings_per_unit_value": 0.0,
                "annual_savings_value": None,
                "confidence_level": 0.6,
                "risk_level": "medium_risk",
                "risk_rationale": "Requires transaction-chain substantiation and transfer pricing controls.",
                "required_actions": [
                    "Trace manufacturer, middleman, and importer invoice chain.",
                    "Validate related-party pricing documentation.",
                ],
                "documentation_required": [
                    "Manufacturer invoice",
                    "Middleman invoice",
                    "Transfer pricing documentation",
                ],
                "implementation_difficulty": "medium",
            }
        )

    if len(bom_items) >= 6:
        strategies.append(
            {
                "strategy_type": "duty_drawback",
                "human_summary": "Duty drawback may recover duties when imported inputs are re-exported in finished goods.",
                "optimized_duty_rate": baseline_rate,
                "savings_per_unit_rate": 0.0,
                "savings_per_unit_value": 0.0,
                "annual_savings_value": None,
                "confidence_level": 0.55,
                "risk_level": "medium_risk",
                "risk_rationale": "Program eligibility depends on strict import-export traceability.",
                "required_actions": [
                    "Map imported components to exported finished-goods lots.",
                    "Establish drawback claim process with reconciliation controls.",
                ],
                "documentation_required": [
                    "Import entries and duty receipts",
                    "Export records",
                    "Manufacturing consumption records",
                ],
                "implementation_difficulty": "medium",
            }
        )

    if material_breakdown:
        boundary = _materials_near_boundary(material_breakdown)
        if boundary and baseline_rate is not None and declared_value_per_unit is not None:
            estimated_rate_delta = min(1.5, baseline_rate * 0.3)
            savings_value = estimated_rate_delta / 100.0 * declared_value_per_unit
            annual_savings = savings_value * annual_volume if annual_volume is not None else None
            strategies.append(
                {
                    "strategy_type": "product_modification",
                    "human_summary": (
                        f"Material mix is near a classification boundary ({boundary[0]} vs {boundary[1]}); "
                        "small composition changes may unlock a lower-duty lane."
                    ),
                    "optimized_duty_rate": max(0.0, baseline_rate - estimated_rate_delta),
                    "savings_per_unit_rate": estimated_rate_delta,
                    "savings_per_unit_value": savings_value,
                    "annual_savings_value": annual_savings,
                    "confidence_level": 0.5,
                    "risk_level": "high_risk",
                    "risk_rationale": "Re-engineering product content can trigger recertification and reclassification review.",
                    "required_actions": [
                        "Run engineering feasibility for alternative material percentages.",
                        "Re-test product compliance after composition change.",
                    ],
                    "documentation_required": [
                        "Updated BOM and material declarations",
                        "Validation test reports",
                    ],
                    "implementation_difficulty": "high",
                }
            )

    return strategies
