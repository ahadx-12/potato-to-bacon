from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import re
from typing import Any, Iterable, Mapping, Sequence

from potatobacon.tariff.duty_calculator import compute_total_duty
from potatobacon.tariff.hts_ingest.usitc_parser import duty_rate_as_float
from potatobacon.tariff.rate_store import RateEntry, get_rate_store
from potatobacon.tariff.auto_classifier import classify_with_thresholds, declared_code_warning

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_DIGIT_RE = re.compile(r"\D")
_AMBIGUITY_HINTS = {"assembly", "kit", "set", "with", "containing", "composite"}

_MATERIAL_TO_CHAPTERS: dict[str, tuple[str, ...]] = {
    "steel": ("72", "73"),
    "iron": ("72", "73"),
    "aluminum": ("76",),
    "aluminium": ("76",),
    "plastics": ("39",),
    "plastic": ("39",),
    "rubber": ("40",),
    "wood": ("44",),
    "textile": tuple(f"{idx:02d}" for idx in range(50, 64)),
    "ceramic": ("69",),
    "glass": ("70",),
    "copper": ("74",),
    "electronics": ("85",),
    "machinery": ("84",),
    "vehicle": ("87",),
    "furniture": ("94",),
}

_MATERIAL_KEYWORDS = {
    "steel",
    "iron",
    "stainless",
    "aluminum",
    "aluminium",
    "copper",
    "brass",
    "plastic",
    "polymer",
    "rubber",
    "wood",
    "textile",
    "glass",
    "ceramic",
    "leather",
}
_PRODUCT_KEYWORDS = {
    "bolt",
    "nut",
    "washer",
    "screw",
    "fastener",
    "wire",
    "mesh",
    "cable",
    "connector",
    "pump",
    "valve",
    "bearing",
    "motor",
    "furniture",
    "chair",
    "desk",
    "shelf",
    "panel",
    "battery",
    "solar",
    "shrimp",
    "honey",
}


@dataclass(frozen=True)
class ReclassificationCandidate:
    hts_code: str
    description: str
    base_rate: float
    total_duty_rate: float
    savings_vs_current: float
    savings_pct: float
    plausibility_score: float
    risk_level: str  # low | medium | high
    risk_rationale: str
    search_type: str  # same_heading | same_chapter | cross_chapter


def _normalize_hts(code: str | None) -> str:
    if not code:
        return ""
    return _DIGIT_RE.sub("", str(code))


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _extract_material_tokens(materials: Sequence[str] | str | None) -> set[str]:
    if materials is None:
        return set()
    if isinstance(materials, str):
        return _tokens(materials)
    output: set[str] = set()
    for item in materials:
        output.update(_tokens(item))
    return output


def _material_to_chapters(material_tokens: Iterable[str]) -> set[str]:
    chapters: set[str] = set()
    for token in material_tokens:
        if token in _MATERIAL_TO_CHAPTERS:
            chapters.update(_MATERIAL_TO_CHAPTERS[token])
    return chapters


@lru_cache(maxsize=1)
def _indexed_entries() -> dict[str, Any]:
    store = get_rate_store()
    by_chapter: dict[str, list[RateEntry]] = {}
    by_heading: dict[str, list[RateEntry]] = {}
    for entry in store.iter_entries():
        digits = _normalize_hts(entry.hts_code)
        if len(digits) < 8:
            continue
        base = duty_rate_as_float(entry.general_rate)
        if base is None:
            continue
        chapter = digits[:2]
        heading = digits[:4]
        by_chapter.setdefault(chapter, []).append(entry)
        by_heading.setdefault(heading, []).append(entry)
    return {"by_chapter": by_chapter, "by_heading": by_heading}


def _rate_entries() -> list[RateEntry]:
    """Compatibility helper used by legacy tests."""
    return get_rate_store().iter_entries()


def _risk_for(search_type: str, current_digits: str, candidate_digits: str) -> tuple[str, str]:
    if search_type == "same_heading":
        return "low", "Reclassification within same heading - low audit risk."
    if search_type == "same_chapter":
        return (
            "medium",
            f"Different heading within Chapter {current_digits[:2]} - may require CBP ruling to confirm.",
        )
    return "high", "Cross-chapter reclassification - recommend obtaining binding ruling from CBP before implementing."


def _candidate_plausibility(
    *,
    product_tokens: set[str],
    material_tokens: set[str],
    entry_description: str,
) -> tuple[float, bool]:
    entry_tokens = _tokens(entry_description)
    if not entry_tokens:
        return 0.0, False
    overlap = len(product_tokens & entry_tokens) / max(1, len(entry_tokens))
    material_match = bool(material_tokens & entry_tokens & _MATERIAL_KEYWORDS)
    product_match = bool(product_tokens & entry_tokens & _PRODUCT_KEYWORDS)
    plausible = material_match or product_match
    if material_match:
        overlap += 0.2
    if product_match:
        overlap += 0.15
    return min(1.0, overlap), plausible


def _material_list(materials: Sequence[str] | str | None) -> list[str]:
    if materials is None:
        return []
    if isinstance(materials, str):
        return [materials]
    return [str(item) for item in materials]


def find_reclassification_candidates(
    hts_code: str,
    description: str,
    materials: Sequence[str] | str,
    value_usd: float,
    origin_country: str,
) -> list[ReclassificationCandidate]:
    """Find legally plausible lower-duty HTS alternatives for a SKU."""

    current_digits = _normalize_hts(hts_code)
    if len(current_digits) < 4:
        return []

    product_tokens = _tokens(description)
    material_tokens = _extract_material_tokens(materials)
    indexed = _indexed_entries()
    by_heading: dict[str, list[RateEntry]] = indexed["by_heading"]
    by_chapter: dict[str, list[RateEntry]] = indexed["by_chapter"]
    chapter = current_digits[:2]
    heading = current_digits[:4]

    try:
        current_duty = compute_total_duty(
            hts_code=hts_code,
            origin_country=origin_country,
            import_country="US",
            declared_value=value_usd,
        )
    except Exception:
        logger.exception("Unable to compute current duty for %s", hts_code)
        return []

    same_heading_entries = list(by_heading.get(heading, []))
    same_chapter_entries = [entry for entry in by_chapter.get(chapter, []) if _normalize_hts(entry.hts_code)[:4] != heading]

    ambiguity = len(_material_list(materials)) >= 2 or bool(_AMBIGUITY_HINTS & product_tokens)
    cross_chapter_entries: list[RateEntry] = []
    if ambiguity:
        for chapter_code in sorted(_material_to_chapters(material_tokens)):
            if chapter_code == chapter:
                continue
            cross_chapter_entries.extend(by_chapter.get(chapter_code, []))

    pool: list[tuple[str, RateEntry]] = []
    pool.extend(("same_heading", entry) for entry in same_heading_entries)
    pool.extend(("same_chapter", entry) for entry in same_chapter_entries)
    pool.extend(("cross_chapter", entry) for entry in cross_chapter_entries)
    if len(pool) > 500:
        pool = pool[:500]

    candidates: list[ReclassificationCandidate] = []
    for search_type, entry in pool:
        candidate_digits = _normalize_hts(entry.hts_code)
        if candidate_digits == current_digits:
            continue
        plausibility_score, plausible = _candidate_plausibility(
            product_tokens=product_tokens,
            material_tokens=material_tokens,
            entry_description=entry.description,
        )
        if not plausible or plausibility_score <= 0.1:
            continue
        try:
            candidate_duty = compute_total_duty(
                hts_code=entry.hts_code,
                origin_country=origin_country,
                import_country="US",
                declared_value=value_usd,
            )
        except Exception:
            continue
        if candidate_duty.total_duty_rate >= current_duty.total_duty_rate:
            continue
        savings_rate = current_duty.total_duty_rate - candidate_duty.total_duty_rate
        risk_level, risk_rationale = _risk_for(search_type, current_digits, candidate_digits)
        candidates.append(
            ReclassificationCandidate(
                hts_code=entry.hts_code,
                description=entry.description,
                base_rate=float(duty_rate_as_float(entry.general_rate) or 0.0),
                total_duty_rate=float(candidate_duty.total_duty_rate),
                savings_vs_current=(savings_rate / 100.0) * value_usd,
                savings_pct=savings_rate,
                plausibility_score=plausibility_score,
                risk_level=risk_level,
                risk_rationale=risk_rationale,
                search_type=search_type,
            )
        )

    risk_order = {"low": 0, "medium": 1, "high": 2}
    candidates.sort(
        key=lambda item: (
            -item.savings_vs_current,
            -item.plausibility_score,
            risk_order[item.risk_level],
            item.hts_code,
        )
    )
    return candidates[:5]


# ---------------------------------------------------------------------------
# Compatibility wrappers used by existing suggest flow
# ---------------------------------------------------------------------------
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
    if not current_hts or not declared_value_per_unit:
        return []
    materials = [material] if material else []
    if material_breakdown:
        for item in material_breakdown:
            mat = str(item.get("material") or "").strip()
            if mat:
                materials.append(mat)
    if baseline_rate is None:
        baseline_rate = 0.0

    # Keep compatibility with older tests that monkeypatch _rate_entries.
    current_digits = _normalize_hts(current_hts)
    chapter = current_digits[:2]
    query_tokens = _tokens(description + " " + " ".join(materials))
    rows: list[ReclassificationCandidate] = []
    for entry in _rate_entries():
        digits = _normalize_hts(entry.hts_code)
        if len(digits) < 4 or digits == current_digits:
            continue
        if digits[:2] != chapter:
            continue
        base = duty_rate_as_float(entry.general_rate)
        if base is None:
            continue
        if float(base) >= baseline_rate:
            continue
        score, plausible = _candidate_plausibility(
            product_tokens=query_tokens,
            material_tokens=_extract_material_tokens(materials),
            entry_description=entry.description,
        )
        if not plausible or score <= 0.1:
            continue
        search_type = "same_heading" if digits[:4] == current_digits[:4] else "same_chapter"
        risk_level, risk_rationale = _risk_for(search_type, current_digits, digits)
        savings_pct = baseline_rate - float(base)
        rows.append(
            ReclassificationCandidate(
                hts_code=entry.hts_code,
                description=entry.description,
                base_rate=float(base),
                total_duty_rate=float(base),
                savings_vs_current=savings_pct / 100.0 * float(declared_value_per_unit),
                savings_pct=savings_pct,
                plausibility_score=score,
                risk_level=risk_level,
                risk_rationale=risk_rationale,
                search_type=search_type,
            )
        )

    rows.sort(key=lambda item: (-item.savings_vs_current, -item.plausibility_score, item.hts_code))
    mapped: list[dict[str, Any]] = []
    risk_map = {"low": "low_risk", "medium": "medium_risk", "high": "high_risk"}
    for row in rows[:top_k]:
        annual = row.savings_vs_current * annual_volume if annual_volume is not None else None
        mapped.append(
            {
                "strategy_type": "reclassification",
                "from_hts": current_hts,
                "to_hts": row.hts_code,
                "candidate_description": row.description,
                "optimized_duty_rate": row.total_duty_rate,
                "savings_per_unit_rate": row.savings_pct,
                "savings_per_unit_value": row.savings_vs_current,
                "annual_savings_value": annual,
                "plausibility_score": round(row.plausibility_score, 4),
                "confidence_level": round(row.plausibility_score, 4),
                "risk_level": risk_map[row.risk_level],
                "risk_rationale": row.risk_rationale,
                "required_actions": ["Validate legal basis and obtain broker/ruling support."],
                "documentation_required": ["BOM, product datasheet, and classification memo"],
                "implementation_difficulty": "medium" if row.risk_level == "medium" else row.risk_level,
            }
        )
    return mapped


def build_advisory_strategies(
    *,
    origin_country: str | None,
    bom_items: Sequence[Mapping[str, Any]],
    baseline_rate: float | None,
    declared_value_per_unit: float | None,
    annual_volume: int | None,
    material_breakdown: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    origin = (origin_country or "").upper()
    baseline = float(baseline_rate or 0.0)
    value = float(declared_value_per_unit or 0.0)
    rows: list[dict[str, Any]] = []

    if origin in {"CN", "VN", "BD"} and baseline > 0:
        min_s = baseline / 100.0 * value * 0.15
        max_s = baseline / 100.0 * value * 0.30
        rows.append(
            {
                "strategy_type": "first_sale_valuation",
                "human_summary": (
                    "If sourced through a middleman, first-sale valuation may reduce dutiable value by 15-30%."
                ),
                "optimized_duty_rate": baseline,
                "savings_per_unit_rate": 0.0,
                "savings_per_unit_value": (min_s + max_s) / 2.0,
                "annual_savings_value": ((min_s + max_s) / 2.0) * annual_volume if annual_volume is not None else None,
                "risk_level": "medium_risk",
                "risk_rationale": "Requires auditable first-sale documentation chain.",
                "confidence_level": 0.6,
                "required_actions": ["Collect manufacturer-to-middleman and middleman-to-importer invoices."],
                "documentation_required": ["Commercial invoice chain", "Transfer-pricing support"],
                "implementation_difficulty": "medium",
            }
        )

    if len(bom_items) >= 5 and baseline > 0 and annual_volume is not None:
        annual_recovery = baseline / 100.0 * value * annual_volume * 0.30 * 0.99
        rows.append(
            {
                "strategy_type": "duty_drawback",
                "human_summary": "If finished goods are re-exported, duty drawback may recover up to 99% of paid duties.",
                "optimized_duty_rate": baseline,
                "savings_per_unit_rate": 0.0,
                "savings_per_unit_value": annual_recovery / max(annual_volume, 1),
                "annual_savings_value": annual_recovery,
                "risk_level": "medium_risk",
                "risk_rationale": "Recovery depends on strict import-export traceability and claim compliance.",
                "confidence_level": 0.55,
                "required_actions": ["Establish drawback traceability and claim process."],
                "documentation_required": ["Import entries", "Export records", "Manufacturing records"],
                "implementation_difficulty": "medium",
            }
        )

    if material_breakdown and baseline > 0:
        shares: list[tuple[str, float]] = []
        for item in material_breakdown:
            material_name = str(item.get("material") or "").strip()
            try:
                share = float(item.get("percent_by_weight") or 0.0)
            except (TypeError, ValueError):
                share = 0.0
            if material_name and share > 0:
                shares.append((material_name, share))
        shares.sort(key=lambda part: part[1], reverse=True)
        if len(shares) >= 2:
            top_material, top_pct = shares[0]
            alt_material, alt_pct = shares[1]
            if top_pct <= 65.0 and abs(top_pct - alt_pct) <= 20.0:
                estimated_delta_rate = min(2.0, baseline * 0.25)
                per_unit = estimated_delta_rate / 100.0 * value
                rows.append(
                    {
                        "strategy_type": "product_modification",
                        "human_summary": (
                            f"Material split is near a classification boundary ({top_pct:.0f}% {top_material} / "
                            f"{alt_pct:.0f}% {alt_material}). Adjusting the dominant material may support "
                            "a lower-duty chapter."
                        ),
                        "optimized_duty_rate": max(0.0, baseline - estimated_delta_rate),
                        "savings_per_unit_rate": estimated_delta_rate,
                        "savings_per_unit_value": per_unit,
                        "annual_savings_value": per_unit * annual_volume if annual_volume is not None else None,
                        "risk_level": "high_risk",
                        "risk_rationale": "Requires product redesign and fresh classification analysis.",
                        "confidence_level": 0.45,
                        "required_actions": ["Evaluate BOM composition changes and re-test product performance."],
                        "documentation_required": ["Updated BOM", "Engineering change order", "Classification memo"],
                        "implementation_difficulty": "high",
                    }
                )
    return rows


def build_auto_classification_payload(
    *,
    description: str,
    material: str = "",
    intended_use: str = "",
    declared_hts: str | None = None,
    confidence_threshold: float = 0.62,
) -> dict[str, Any]:
    """Compatibility helper used by suggest.py and legacy tests."""

    query_tokens = _tokens(" ".join([description, material, intended_use]))
    material_tokens = _extract_material_tokens(material)
    scored: list[dict[str, Any]] = []
    for entry in _rate_entries():
        base = duty_rate_as_float(entry.general_rate)
        if base is None:
            continue
        score, plausible = _candidate_plausibility(
            product_tokens=query_tokens,
            material_tokens=material_tokens,
            entry_description=entry.description,
        )
        if not plausible:
            continue
        scored.append(
            {
                "hts_code": entry.hts_code,
                "description": entry.description,
                "base_rate": float(base),
                "confidence": round(score, 4),
            }
        )
    scored.sort(key=lambda item: (-item["confidence"], item["hts_code"]))
    candidates = scored[:3]
    top_conf = float(candidates[0]["confidence"]) if candidates else 0.0
    if declared_hts:
        warning = declared_code_warning(
            declared_hts_code=declared_hts,
            description=description,
            materials=material,
            weight_kg=None,
            value_usd=None,
            intended_use=intended_use,
        )
        classification = "provided"
        hts_source = "declared"
        selected = declared_hts
        mismatch_flag = warning is not None
        review_reason = warning
    else:
        if top_conf >= max(0.7, confidence_threshold):
            classification = "auto"
            hts_source = "auto_classified"
            selected = candidates[0]["hts_code"] if candidates else None
            mismatch_flag = False
            review_reason = None
        elif top_conf >= 0.4:
            classification = "auto"
            hts_source = "auto_classified"
            selected = candidates[0]["hts_code"] if candidates else None
            mismatch_flag = False
            review_reason = "Auto-classified with low confidence; manual review recommended."
        else:
            classification = "manual_review"
            hts_source = "unresolved"
            selected = None
            mismatch_flag = False
            review_reason = "No candidate exceeded the minimum confidence threshold."

    return {
        "classification": classification,
        "hts_source": hts_source,
        "selected_hts_code": selected,
        "declared_hts_code": declared_hts,
        "confidence": top_conf,
        "needs_manual_review": bool(review_reason) or mismatch_flag,
        "review_reason": review_reason,
        "mismatch_flag": mismatch_flag,
        "alternatives": [
            {
                "hts_code": item["hts_code"],
                "description": item["description"],
                "duty_rate": item["base_rate"],
                "confidence": item["confidence"],
            }
            for item in candidates
        ],
    }
