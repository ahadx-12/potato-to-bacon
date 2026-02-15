from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import re
from typing import Any, Iterable, Sequence

from potatobacon.tariff.hts_ingest.usitc_parser import duty_rate_as_float
from potatobacon.tariff.rate_store import RateEntry, get_rate_store

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_DIGIT_RE = re.compile(r"\D")


@dataclass(frozen=True)
class ClassificationCandidate:
    hts_code: str
    description: str
    confidence: float
    match_reasons: list[str]
    base_rate: float
    chapter: int


_MATERIAL_CHAPTERS: dict[str, tuple[int, ...]] = {
    "steel": (72, 73),
    "iron": (72, 73),
    "aluminum": (76,),
    "aluminium": (76,),
    "copper": (74,),
    "plastic": (39,),
    "polymer": (39,),
    "polyethylene": (39,),
    "pvc": (39,),
    "polypropylene": (39,),
    "rubber": (40,),
    "silicone": (40,),
    "wood": (44,),
    "timber": (44,),
    "plywood": (44,),
    "cotton": tuple(range(50, 64)),
    "polyester": tuple(range(50, 64)),
    "nylon": tuple(range(50, 64)),
    "silk": tuple(range(50, 64)),
    "wool": tuple(range(50, 64)),
    "textile": tuple(range(50, 64)),
    "fabric": tuple(range(50, 64)),
    "glass": (70,),
    "ceramic": (69,),
    "porcelain": (69,),
    "paper": (48,),
    "cardboard": (48,),
    "leather": (41, 42),
    "brass": (74,),
    "titanium": (81,),
    "foam": (39, 40),
    "electronics": (85,),
    "battery": (85,),
}

_PRODUCT_CHAPTERS: dict[str, tuple[int, ...]] = {
    "motor": (84,),
    "engine": (84,),
    "pump": (84,),
    "valve": (84,),
    "bearing": (84,),
    "gear": (84,),
    "wire": (85,),
    "cable": (85,),
    "connector": (85,),
    "adapter": (85,),
    "switch": (85,),
    "led": (85,),
    "battery": (85,),
    "phone": (85,),
    "smartphone": (85,),
    "inverter": (85,),
    "controller": (85,),
    "circuit": (85,),
    "vehicle": (87,),
    "brake": (87,),
    "wheel": (87,),
    "axle": (87,),
    "bumper": (87,),
    "headlamp": (87,),
    "wiper": (87,),
    "furniture": (94,),
    "chair": (94,),
    "desk": (94,),
    "shelf": (94,),
    "mattress": (94,),
    "sofa": (94,),
    "backpack": (42,),
    "case": (42, 85),
    "glove": (40,),
    "gloves": (40,),
    "toy": (95,),
    "game": (95,),
    "tool": (82,),
    "drill": (82,),
    "saw": (82,),
    "wrench": (82,),
    "bolt": (73,),
    "nut": (73,),
    "screw": (73,),
    "fastener": (73,),
    "washer": (73,),
    "rivet": (73,),
    "pipe": (73,),
    "tube": (73,),
    "fitting": (73,),
    "filter": (84,),
    "solar": (85,),
    "photovoltaic": (85,),
    "food": tuple(range(1, 25)),
    "fruit": tuple(range(1, 25)),
    "vegetable": tuple(range(1, 25)),
    "meat": tuple(range(1, 25)),
    "fish": tuple(range(1, 25)),
    "dairy": tuple(range(1, 25)),
    "chemical": tuple(range(28, 39)),
    "resin": tuple(range(28, 39)),
    "adhesive": tuple(range(28, 39)),
    "solvent": tuple(range(28, 39)),
    "acid": tuple(range(28, 39)),
    "paint": (32,),
    "ink": (32,),
    "pigment": (32,),
    "soap": (34,),
    "detergent": (34,),
    "cosmetic": (34,),
    "medicine": (30,),
    "pharmaceutical": (30,),
    "fertilizer": (31,),
}


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _normalize_hts(code: str) -> str:
    return _DIGIT_RE.sub("", str(code))


@lru_cache(maxsize=1)
def _entries() -> list[RateEntry]:
    return get_rate_store().iter_entries()


def _infer_candidate_chapters(description_tokens: set[str], material_tokens: set[str]) -> tuple[dict[int, float], set[int]]:
    weights: dict[int, float] = {}
    product_chapters: set[int] = set()
    for token in material_tokens:
        for chapter in _MATERIAL_CHAPTERS.get(token, ()):
            weights[chapter] = weights.get(chapter, 0.0) + 1.0
    for token in description_tokens:
        for chapter in _PRODUCT_CHAPTERS.get(token, ()):
            weights[chapter] = weights.get(chapter, 0.0) + 1.6
            product_chapters.add(chapter)
    return weights, product_chapters


def _specificity_rank(hts_code: str) -> int:
    digits = _normalize_hts(hts_code)
    return len(digits)


def _heuristic_category_boost(
    *,
    chapter: int,
    weight_kg: float | None,
    value_usd: float | None,
) -> float:
    if weight_kg is None or value_usd is None:
        return 0.0
    if weight_kg > 8 and value_usd < 20 and chapter in {39, 40, 72, 73, 74, 76}:
        return 0.1
    if weight_kg < 3 and value_usd > 80 and chapter in {84, 85, 90}:
        return 0.1
    return 0.0


def classify_product(
    description: str,
    materials: Sequence[str] | str,
    weight_kg: float | None,
    value_usd: float | None,
    intended_use: str | None = None,
) -> list[ClassificationCandidate]:
    """Return top 3 HTS candidates ranked by deterministic confidence."""

    if isinstance(materials, str):
        materials_text = materials
    else:
        materials_text = " ".join(str(item) for item in materials)
    query_text = " ".join(part for part in [description, materials_text, intended_use or ""] if part)

    description_tokens = _tokens(query_text)
    material_tokens = _tokens(materials_text)
    chapter_weights, product_chapters = _infer_candidate_chapters(description_tokens, material_tokens)
    candidate_chapters = set(product_chapters) if product_chapters else set(chapter_weights.keys())
    if candidate_chapters:
        ranked = sorted(candidate_chapters, key=lambda ch: (-chapter_weights.get(ch, 0.0), ch))
        candidate_chapters = set(ranked[:8])

    scored: list[tuple[float, int, str, ClassificationCandidate]] = []
    chapter_fallback: list[tuple[float, int, str, ClassificationCandidate]] = []
    for entry in _entries():
        digits = _normalize_hts(entry.hts_code)
        if len(digits) < 4:
            continue
        chapter = int(digits[:2])
        if candidate_chapters and chapter not in candidate_chapters:
            continue

        base_rate = duty_rate_as_float(entry.general_rate)
        if base_rate is None:
            continue

        entry_tokens = _tokens(entry.description)
        if not entry_tokens:
            continue
        overlap = len(description_tokens & entry_tokens) / max(1, len(entry_tokens))

        reasons: list[str] = []
        material_match = sorted((material_tokens & entry_tokens) & set(_MATERIAL_CHAPTERS.keys()))
        product_match = sorted((description_tokens & entry_tokens) & set(_PRODUCT_CHAPTERS.keys()))
        score = overlap
        if material_match:
            score += 0.3
            reasons.append(f"material match: {material_match[0]}")
        if product_match:
            score += 0.2
            reasons.append(f"product keyword: {product_match[0]}")

        score += _heuristic_category_boost(chapter=chapter, weight_kg=weight_kg, value_usd=value_usd)
        score += min(0.25, 0.04 * chapter_weights.get(chapter, 0.0))
        score = min(1.0, score)
        if overlap > 0 and not reasons:
            reasons.append("description token overlap")
        elif overlap <= 0 and not reasons:
            reasons.append("chapter pre-filter match")

        candidate = ClassificationCandidate(
            hts_code=entry.hts_code,
            description=entry.description,
            confidence=score,
            match_reasons=reasons,
            base_rate=float(base_rate),
            chapter=chapter,
        )
        row = (score, _specificity_rank(entry.hts_code), entry.hts_code, candidate)
        if overlap > 0:
            scored.append(row)
        else:
            # Deterministic fallback for sparse descriptions/chapters.
            fallback_score = 0.15 + (0.1 if material_match else 0.0) + (0.1 if product_match else 0.0)
            candidate = ClassificationCandidate(
                hts_code=entry.hts_code,
                description=entry.description,
                confidence=min(0.39, fallback_score),
                match_reasons=candidate.match_reasons,
                base_rate=candidate.base_rate,
                chapter=candidate.chapter,
            )
            chapter_fallback.append(
                (candidate.confidence, _specificity_rank(entry.hts_code), entry.hts_code, candidate)
            )

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    if len(scored) < 3:
        chapter_fallback.sort(key=lambda item: (-item[0], -item[1], item[2]))
        scored.extend(chapter_fallback[: max(0, 3 - len(scored))])
    return [item[3] for item in scored[:3]]


def classify_with_thresholds(
    description: str,
    materials: Sequence[str] | str,
    weight_kg: float | None,
    value_usd: float | None,
    intended_use: str | None = None,
) -> dict[str, Any]:
    candidates = classify_product(description, materials, weight_kg, value_usd, intended_use)
    top = candidates[0] if candidates else None
    if top is None:
        return {
            "hts_source": "manual_classification_required",
            "review_required": True,
            "selected_hts_code": None,
            "candidates": [],
        }
    if top.confidence >= 0.7:
        source = "auto_classified"
        review = False
    elif top.confidence >= 0.4:
        source = "auto_classified_low_confidence"
        review = True
    else:
        source = "manual_classification_required"
        review = True
    return {
        "hts_source": source,
        "review_required": review,
        "selected_hts_code": top.hts_code if source != "manual_classification_required" else None,
        "top_confidence": top.confidence,
        "candidates": [
            {
                "hts_code": c.hts_code,
                "description": c.description,
                "confidence": c.confidence,
                "match_reasons": c.match_reasons,
                "base_rate": c.base_rate,
                "chapter": c.chapter,
            }
            for c in candidates
        ],
    }


def declared_code_warning(
    *,
    declared_hts_code: str,
    description: str,
    materials: Sequence[str] | str,
    weight_kg: float | None,
    value_usd: float | None,
    intended_use: str | None = None,
) -> str | None:
    candidates = classify_product(description, materials, weight_kg, value_usd, intended_use)
    if not candidates:
        return None
    top = candidates[0]
    declared_digits = _normalize_hts(declared_hts_code)
    if len(declared_digits) < 2:
        return None
    declared_chapter = int(declared_digits[:2])
    if declared_chapter != top.chapter:
        return (
            f"Declared HTS {declared_hts_code} (Ch.{declared_chapter}) may be inconsistent with product "
            f"description suggesting Ch.{top.chapter}. Review recommended."
        )
    return None
