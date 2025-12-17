"""Deterministic parser that maps free text into product specs and facts."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import fact_compiler
from .product_schema import (
    MaterialBreakdown,
    ProductCategory,
    ProductSpecModel,
    SurfaceCoverage,
)
from .models import FactEvidenceModel, TextEvidenceModel


def _capture_snippet(text: str, keyword: str, source: str) -> TextEvidenceModel:
    lowered = text.lower()
    idx = lowered.find(keyword)
    start = idx if idx >= 0 else None
    end = start + len(keyword) if start is not None else None
    if start is not None:
        window_start = max(0, start - 10)
        window_end = min(len(text), end + 10) if end is not None else len(text)
        snippet = text[window_start:window_end]
    else:
        snippet = keyword
    return TextEvidenceModel(source=source, snippet=snippet.strip(), start=start, end=end)


def _detect_keywords(text: str, keywords: List[str]) -> List[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw in lowered]


def extract_product_spec(description: str, bom_text: str | None) -> Tuple[ProductSpecModel, List[TextEvidenceModel]]:
    """Deterministically infer a :class:`ProductSpecModel` and extraction evidence."""

    evidence: List[TextEvidenceModel] = []
    desc_lower = description.lower()
    bom_lower = (bom_text or "").lower()

    category_hits = {
        ProductCategory.FOOTWEAR: ["shoe", "sneaker", "boot", "footwear"],
        ProductCategory.FASTENER: ["bolt", "fastener", "screw"],
        ProductCategory.ELECTRONICS: ["circuit", "board", "chip", "sensor", "electronics"],
    }

    detected_category = ProductCategory.OTHER
    for category, keywords in category_hits.items():
        hits = _detect_keywords(desc_lower, keywords) + _detect_keywords(bom_lower, keywords)
        if hits:
            detected_category = category
            hit = hits[0]
            if hit in desc_lower:
                evidence.append(_capture_snippet(description, hit, "description"))
            else:
                evidence.append(_capture_snippet(bom_text or "", hit, "bom_text"))
            break

    materials: List[MaterialBreakdown] = []
    material_keywords = {
        "steel": "steel",
        "aluminum": "aluminum",
        "aluminium": "aluminum",
        "canvas": "textile",
        "textile": "textile",
        "rubber": "rubber",
        "plastic": "plastic",
        "plastics": "plastic",
    }

    for keyword, material in material_keywords.items():
        if keyword in desc_lower:
            evidence.append(_capture_snippet(description, keyword, "description"))
            component = "upper" if material == "textile" else "sole" if material == "rubber" else "component"
            materials.append(MaterialBreakdown(component=component, material=material))
        elif keyword in bom_lower:
            evidence.append(_capture_snippet(bom_text or "", keyword, "bom_text"))
            materials.append(MaterialBreakdown(component="component", material=material))

    surface_coverage: List[SurfaceCoverage] = []
    if "rubber sole" in desc_lower or "rubber sole" in bom_lower:
        surface_coverage.append(SurfaceCoverage(material="rubber", percent_coverage=80.0))
    if "felt overlay" in desc_lower or "felt overlay" in bom_lower:
        evidence.append(_capture_snippet(description, "felt overlay", "description"))
        surface_coverage.append(
            SurfaceCoverage(material="textile", percent_coverage=60.0, coating_type="felt overlay")
        )
    if ">50%" in desc_lower or "greater than 50" in desc_lower:
        surface_coverage.append(SurfaceCoverage(material="textile", percent_coverage=60.0))

    spec = ProductSpecModel(
        product_category=detected_category,
        materials=materials,
        surface_coverage=surface_coverage,
    )
    return spec, evidence


def _fact_confidence(has_evidence: bool, unknown: bool = False) -> float:
    if unknown:
        return 0.2
    return 0.9 if has_evidence else 0.4


def compile_facts_with_evidence(
    spec: ProductSpecModel, description: str, bom_text: str | None
) -> Tuple[Dict[str, Any], List[FactEvidenceModel]]:
    """Compile facts from a product spec while attaching text evidence."""

    facts, derived_evidence = fact_compiler.compile_facts(spec)
    fact_evidence: List[FactEvidenceModel] = []
    text_sources = {"description": description, "bom_text": bom_text or ""}
    combined_text = f"{description} {bom_text or ''}".lower()

    keyword_map: Dict[str, List[str]] = {
        "product_category": [spec.product_category.value],
        "product_type_chassis_bolt": ["bolt"],
        "material_steel": ["steel"],
        "material_aluminum": ["aluminum", "aluminium"],
        "material_textile": ["textile", "canvas", "felt"],
        "material_rubber": ["rubber"],
    }

    # heuristic fastener fact if bolt detected
    if "bolt" in combined_text and spec.product_category == ProductCategory.FASTENER:
        facts.setdefault("product_type_chassis_bolt", True)

    for fact_key, value in facts.items():
        snippets: List[TextEvidenceModel] = []
        keywords = keyword_map.get(fact_key, [])
        for keyword in keywords:
            for source, raw_text in text_sources.items():
                if keyword and keyword in raw_text.lower():
                    snippets.append(_capture_snippet(raw_text, keyword, source))
                    break
            if snippets:
                break

        confidence = _fact_confidence(bool(snippets), spec.product_category == ProductCategory.OTHER)
        fact_evidence.append(
            FactEvidenceModel(
                fact_key=fact_key,
                value=value,
                confidence=confidence,
                evidence=snippets,
                derived_from=["description"] if snippets else [],
            )
        )

    # add evidence for derived facts from compiler that may not have snippets
    derived_map = {item.fact_key: item for item in derived_evidence}
    for fact_item in fact_evidence:
        derived = derived_map.get(fact_item.fact_key)
        if derived:
            fact_item.derived_from = derived.derived_from_fields

    return facts, fact_evidence
