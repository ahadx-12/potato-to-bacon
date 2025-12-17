"""Deterministic parser that maps free text into product specs and facts."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import fact_compiler
from .bom_ingest import bom_aggregate_material_signals, bom_to_text
from .models import FactEvidenceModel, StructuredBOMModel, TextEvidenceModel
from .product_schema import (
    MaterialBreakdown,
    ProductCategory,
    ProductSpecModel,
    SurfaceCoverage,
)


def _capture_snippet(text: str, keyword: str, source: str, window: int = 40) -> TextEvidenceModel:
    lowered = text.lower()
    idx = lowered.find(keyword.lower())
    start = idx if idx >= 0 else None
    end = start + len(keyword) if start is not None else None
    if start is not None:
        window_start = max(0, start - window)
        window_end = min(len(text), end + window) if end is not None else len(text)
        snippet = text[window_start:window_end]
    else:
        snippet = keyword
    return TextEvidenceModel(source=source, snippet=snippet.strip(), start=start, end=end)


def _detect_keywords(text: str, keywords: List[str]) -> List[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw in lowered]


def _text_sources(
    description: str,
    bom_text: str | None,
    bom_structured: StructuredBOMModel | None,
) -> List[Tuple[str, str]]:
    sources: List[Tuple[str, str]] = [("description", description)]
    if bom_structured:
        sources.append(("bom_json", bom_to_text(bom_structured)))
    elif bom_text:
        sources.append(("bom_text", bom_text))
    return sources


def extract_product_spec(
    description: str,
    bom_text: str | None,
    bom_structured: StructuredBOMModel | None = None,
    origin_country: str | None = None,
    export_country: str | None = None,
    import_country: str | None = None,
) -> Tuple[ProductSpecModel, List[TextEvidenceModel]]:
    """Deterministically infer a :class:`ProductSpecModel` and extraction evidence."""

    evidence: List[TextEvidenceModel] = []
    sources = _text_sources(description, bom_text, bom_structured)
    combined_lower = " ".join(text for _, text in sources).lower()

    category_hits = {
        ProductCategory.FOOTWEAR: ["shoe", "sneaker", "boot", "footwear"],
        ProductCategory.FASTENER: ["bolt", "fastener", "screw"],
        ProductCategory.ELECTRONICS: [
            "circuit",
            "board",
            "chip",
            "sensor",
            "electronics",
            "pcb",
            "printed circuit",
        ],
        ProductCategory.APPAREL_TEXTILE: [
            "shirt",
            "jacket",
            "hoodie",
            "pants",
            "leggings",
            "socks",
            "gloves",
            "textile",
            "apparel",
        ],
    }

    detected_category = ProductCategory.OTHER
    for category, keywords in category_hits.items():
        for source_name, source_text in sources:
            hits = _detect_keywords(source_text.lower(), keywords)
            if hits:
                detected_category = category
                evidence.append(_capture_snippet(source_text, hits[0], source_name))
                break
        if detected_category != ProductCategory.OTHER:
            break

    materials: List[MaterialBreakdown] = []
    electronics_flags = {
        "has_pcb": False,
        "is_cable_or_connector": False,
        "is_enclosure_or_housing": False,
        "contains_battery": False,
    }
    apparel_flags = {
        "is_knit": False,
        "is_woven": False,
        "has_coating_or_lamination": False,
    }
    fiber_pcts = {"fiber_cotton_pct": None, "fiber_polyester_pct": None, "fiber_nylon_pct": None}

    electronics_keywords = {
        "has_pcb": ["pcb", "circuit board", "printed circuit", "microcontroller", "controller board"],
        "is_enclosure_or_housing": ["enclosure", "housing", "casing"],
        "is_cable_or_connector": ["cable", "connector", "usb", "hdmi"],
        "contains_battery": ["lithium", "battery pack", "cell", "battery"],
    }
    apparel_keywords = {
        "is_knit": ["knit"],
        "is_woven": ["woven"],
        "has_coating_or_lamination": ["coated", "laminated", "waterproof", "membrane"],
    }
    fiber_keywords = {
        "fiber_cotton_pct": ["cotton"],
        "fiber_polyester_pct": ["polyester", "spandex", "elastane"],
        "fiber_nylon_pct": ["nylon"],
    }

    material_keywords = {
        "steel": "steel",
        "aluminum": "aluminum",
        "aluminium": "aluminum",
        "canvas": "textile",
        "textile": "textile",
        "rubber": "rubber",
        "plastic": "plastic",
        "plastics": "plastic",
        "abs": "plastic",
        "pc": "plastic",
        "cotton": "textile",
        "polyester": "textile",
        "nylon": "textile",
    }

    for source_name, source_text in sources:
        lowered = source_text.lower()
        for key, keywords in electronics_keywords.items():
            hits = _detect_keywords(lowered, keywords)
            if hits:
                electronics_flags[key] = True
                evidence.append(_capture_snippet(source_text, hits[0], source_name))
        for key, keywords in apparel_keywords.items():
            hits = _detect_keywords(lowered, keywords)
            if hits:
                apparel_flags[key] = True
                evidence.append(_capture_snippet(source_text, hits[0], source_name))
        for key, keywords in fiber_keywords.items():
            hits = _detect_keywords(lowered, keywords)
            if hits and fiber_pcts[key] is None:
                fiber_pcts[key] = 60.0
                evidence.append(_capture_snippet(source_text, hits[0], source_name))
        for keyword, material in material_keywords.items():
            if keyword in lowered:
                evidence.append(_capture_snippet(source_text, keyword, source_name))
                component = "upper" if material == "textile" else "sole" if material == "rubber" else "component"
                materials.append(MaterialBreakdown(component=component, material=material))

    surface_coverage: List[SurfaceCoverage] = []
    if "rubber sole" in combined_lower:
        surface_coverage.append(SurfaceCoverage(material="rubber", percent_coverage=80.0))
    if "felt overlay" in combined_lower:
        evidence.append(_capture_snippet(description, "felt overlay", "description"))
        surface_coverage.append(
            SurfaceCoverage(material="textile", percent_coverage=60.0, coating_type="felt overlay")
        )
    if ">50%" in combined_lower or "greater than 50" in combined_lower:
        surface_coverage.append(SurfaceCoverage(material="textile", percent_coverage=60.0))

    spec = ProductSpecModel(
        product_category=detected_category,
        materials=materials,
        surface_coverage=surface_coverage,
        has_pcb=electronics_flags["has_pcb"],
        is_cable_or_connector=electronics_flags["is_cable_or_connector"],
        is_enclosure_or_housing=electronics_flags["is_enclosure_or_housing"],
        contains_battery=electronics_flags["contains_battery"],
        is_knit=apparel_flags["is_knit"],
        is_woven=apparel_flags["is_woven"],
        has_coating_or_lamination=apparel_flags["has_coating_or_lamination"],
        fiber_cotton_pct=fiber_pcts["fiber_cotton_pct"],
        fiber_polyester_pct=fiber_pcts["fiber_polyester_pct"],
        fiber_nylon_pct=fiber_pcts["fiber_nylon_pct"],
        origin_country=origin_country,
        export_country=export_country,
        import_country=import_country,
    )
    return spec, evidence


def _fact_confidence(fact_key: str, has_evidence: bool, category: ProductCategory) -> float:
    if fact_key == "product_category":
        return 0.6 if has_evidence else 0.4
    if category == ProductCategory.OTHER:
        return 0.4
    return 0.9 if has_evidence else 0.4


def compile_facts_with_evidence(
    spec: ProductSpecModel,
    description: str,
    bom_text: str | None,
    bom_structured: StructuredBOMModel | None = None,
    include_fact_evidence: bool = True,
) -> Tuple[Dict[str, Any], List[FactEvidenceModel]]:
    """Compile facts from a product spec while attaching text evidence."""

    bom_signals = bom_aggregate_material_signals(bom_structured) if bom_structured else None
    facts, derived_evidence = fact_compiler.compile_facts(spec, bom_signals=bom_signals)
    fact_evidence: List[FactEvidenceModel] = []

    sources = _text_sources(description, bom_text, bom_structured)
    keyword_map: Dict[str, List[str]] = {
        "product_category": [spec.product_category.value],
        "product_type_chassis_bolt": ["bolt"],
        "material_steel": ["steel"],
        "material_aluminum": ["aluminum", "aluminium"],
        "material_plastic": ["plastic", "abs"],
        "material_textile": ["textile", "canvas", "felt"],
        "material_rubber": ["rubber"],
        "contains_pcb": ["pcb", "circuit board"],
        "electronics_enclosure": ["enclosure", "housing", "casing"],
        "electronics_cable_or_connector": ["cable", "connector", "usb", "hdmi"],
        "contains_battery": ["battery", "cell", "lithium"],
        "textile_knit": ["knit"],
        "textile_woven": ["woven"],
        "fiber_cotton_dominant": ["cotton"],
        "fiber_polyester_dominant": ["polyester"],
        "has_coating_or_lamination": ["coated", "laminated", "membrane"],
        "ad_cvd_possible": ["steel", "fastener"],
    }

    for fact_key, value in facts.items():
        if not include_fact_evidence:
            fact_evidence.append(
                FactEvidenceModel(
                    fact_key=fact_key,
                    value=value,
                    confidence=_fact_confidence(fact_key, False, spec.product_category),
                    evidence=[],
                    derived_from=[],
                )
            )
            continue

        snippets: List[TextEvidenceModel] = []
        keywords = keyword_map.get(fact_key, [])
        for keyword in keywords:
            for source, raw_text in sources:
                if keyword and keyword.lower() in raw_text.lower():
                    snippets.append(_capture_snippet(raw_text, keyword, source))
                    break
            if snippets:
                break

        if not snippets and fact_key.startswith("origin_country_"):
            snippets.append(TextEvidenceModel(source="payload", snippet=f"{fact_key}=True"))

        if (
            not snippets
            and bom_structured is not None
            and (
                fact_key.startswith("material_")
                or fact_key.startswith("product_type_")
                or fact_key.startswith("electronics_")
                or fact_key in {"contains_pcb", "contains_battery"}
            )
        ):
            snippets.append(TextEvidenceModel(source="bom_json", snippet=f"bom_json evidence for {fact_key}"))

        confidence = _fact_confidence(fact_key, bool(snippets), spec.product_category)
        risk_reason = None
        if not snippets:
            confidence = min(confidence, 0.4)
            risk_reason = "Key fact lacks evidence snippet; requires verification."

        fact_evidence.append(
            FactEvidenceModel(
                fact_key=fact_key,
                value=value,
                confidence=confidence,
                evidence=snippets,
                derived_from=[],
                risk_reason=risk_reason,
            )
        )

    derived_map = {item.fact_key: item for item in derived_evidence}
    for fact_item in fact_evidence:
        derived = derived_map.get(fact_item.fact_key)
        if derived:
            fact_item.derived_from = derived.derived_from_fields

    fact_evidence.sort(
        key=lambda item: (
            item.fact_key,
            str(item.value),
            item.confidence,
            len(item.evidence),
        )
    )
    return facts, fact_evidence
