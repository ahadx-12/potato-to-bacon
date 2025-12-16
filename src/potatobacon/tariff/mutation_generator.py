"""Generate structured mutation candidates from a product spec or free text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .product_schema import ProductCategory, ProductSpecModel


@dataclass
class MutationCandidate:
    """Candidate redesign captured as a patch plus descriptive metadata."""

    mutation_patch: Dict[str, object]
    human_description: str
    expected_tradeoffs: str
    risk_hint: str


_DEFENSIBLE_FASTENER_MUTATIONS = [
    (
        "material",
        "aluminum",
        "Switch steel bolt to aluminum alloy 6061 to lower duty bracket.",
        "May reduce tensile strength; verify torque requirements and corrosion mitigation.",
        "Common substitution for weight-sensitive assemblies.",
    ),
    (
        "material",
        "coated_carbon_steel",
        "Swap stainless steel to coated carbon steel for alternative classification.",
        "Requires coating durability testing and potential anti-corrosion treatments.",
        "Moderate novelty; coating evidence needed.",
    ),
]


_FOOTWEAR_MUTATIONS = [
    (
        "surface_coverage",
        {
            "material": "textile",
            "percent_coverage": 55.0,
            "coating_type": "felt overlay",
        },
        "Increase textile surface coverage above 50% with felt overlay.",
        "Adds material cost and manufacturing step; may change aesthetic.",
        "Coverage threshold near boundary; document measurement method.",
    ),
    (
        "upper_material",
        "textile",
        "Swap upper to majority textile composition.",
        "Potential comfort and durability changes; confirm supplier availability.",
        "Common shift for casual footwear lines.",
    ),
]


_ELECTRONICS_MUTATIONS = [
    (
        "housing_material",
        "plastic",
        "Use plastic housing instead of metal to reduce classification burden.",
        "May impact heat dissipation; validate thermal performance.",
        "Low novelty; standard cost-down pathway.",
    ),
    (
        "assembly_state",
        "kit",
        "Ship as knocked-down kit instead of assembled board.",
        "Requires customer assembly; check regulatory labeling.",
        "Higher scrutiny for functional equivalence.",
    ),
]


_TEXTILE_MUTATIONS = [
    (
        "fiber_mix",
        {"synthetic_percent": 55.0, "natural_percent": 45.0},
        "Adjust fiber mix to cross 50% synthetic threshold.",
        "Sourcing changes required; verify hand-feel and strength.",
        "Threshold-driven; provide lab composition tests.",
    ),
]


@dataclass
class GeneratedMutation:
    candidate: MutationCandidate
    product: ProductSpecModel


_DEFENSIBLE_LIBRARY = {
    ProductCategory.FASTENER: _DEFENSIBLE_FASTENER_MUTATIONS,
    ProductCategory.FOOTWEAR: _FOOTWEAR_MUTATIONS,
    ProductCategory.ELECTRONICS: _ELECTRONICS_MUTATIONS,
    ProductCategory.TEXTILE: _TEXTILE_MUTATIONS,
}


def _select_fastener_candidates(product: ProductSpecModel) -> List[MutationCandidate]:
    candidates: List[MutationCandidate] = []
    materials = {material.material.lower() for material in product.materials}
    for material_key, replacement, desc, tradeoffs, hint in _DEFENSIBLE_FASTENER_MUTATIONS:
        if material_key == "material" and "steel" in materials:
            candidates.append(
                MutationCandidate(
                    mutation_patch={"materials": [{"component": "body", "material": replacement}]},
                    human_description=desc,
                    expected_tradeoffs=tradeoffs,
                    risk_hint=hint,
                )
            )
        elif material_key == "material" and "stainless steel" in materials:
            candidates.append(
                MutationCandidate(
                    mutation_patch={"materials": [{"component": "body", "material": replacement}]},
                    human_description=desc,
                    expected_tradeoffs=tradeoffs,
                    risk_hint=hint,
                )
            )
    return candidates


def _select_category_templates(product: ProductSpecModel) -> List[MutationCandidate]:
    templates = _DEFENSIBLE_LIBRARY.get(product.product_category, [])
    candidates: List[MutationCandidate] = []

    if product.product_category == ProductCategory.FASTENER:
        return _select_fastener_candidates(product)

    for key, value, description, tradeoffs, hint in templates:
        patch: Dict[str, object] = {key: value}
        candidates.append(
            MutationCandidate(
                mutation_patch=patch,
                human_description=description,
                expected_tradeoffs=tradeoffs,
                risk_hint=hint,
            )
        )
    return candidates


def generate_mutation_candidates(product: ProductSpecModel) -> List[MutationCandidate]:
    """Generate deterministic mutation candidates based on product category."""

    return _select_category_templates(product)


# -----------------------------------------------------------------------------
# Free-text profile inference + deterministic mutation generation
# -----------------------------------------------------------------------------


_FOOTWEAR_KEYWORDS = ["shoe", "sneaker", "footwear"]
_FASTENER_KEYWORDS = ["bolt", "fastener", "screw"]


def _detect_keywords(text: str, keyword_list: List[str]) -> List[str]:
    return [kw for kw in keyword_list if kw in text]


def infer_product_profile(description: str, bom_text: str | None) -> Dict[str, Any]:
    """Infer a lightweight product profile from free-text description and BOM."""

    combined = " ".join(filter(None, [description, bom_text or ""])).lower()
    footwear_hits = _detect_keywords(combined, _FOOTWEAR_KEYWORDS)
    fastener_hits = _detect_keywords(combined, _FASTENER_KEYWORDS)

    category = "unknown"
    if footwear_hits:
        category = "footwear"
    elif fastener_hits:
        category = "fastener"

    keywords = sorted(set(footwear_hits + fastener_hits))
    notes: List[str] = []
    if not keywords:
        notes.append("No category keywords detected")

    return {"category": category, "keywords": keywords, "notes": notes}


def baseline_facts_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return baseline tariff facts for a derived profile."""

    category = profile.get("category") if isinstance(profile, dict) else None
    if category == "footwear":
        return {
            "upper_material_textile": True,
            "outer_sole_material_rubber_or_plastics": True,
            "surface_contact_rubber_gt_50": True,
            "surface_contact_textile_gt_50": False,
            "felt_covering_gt_50": False,
        }
    if category == "fastener":
        return {
            "product_type_chassis_bolt": True,
            "material_steel": True,
            "material_aluminum": False,
        }
    return {}


def generate_candidate_mutations(profile: Any) -> List[Any]:
    """Generate deterministic mutations from a free-text profile or product spec."""

    if isinstance(profile, ProductSpecModel):
        return _select_category_templates(profile)

    category = profile.get("category") if isinstance(profile, dict) else None
    if category == "footwear":
        return [
            {"felt_covering_gt_50": True},
            {
                "surface_contact_textile_gt_50": True,
                "surface_contact_rubber_gt_50": False,
            },
        ]
    if category == "fastener":
        return [
            {"material_steel": False, "material_aluminum": True},
            {"material_aluminum": True},
        ]
    return []


def human_summary_for_mutation(profile: Dict[str, Any], mutation: Dict[str, Any]) -> str:
    """Return a human-readable description for a proposed mutation."""

    category = profile.get("category") if isinstance(profile, dict) else None
    if category == "footwear":
        if mutation.get("felt_covering_gt_50"):
            return "Add >50% felt/textile overlay on outsole to make textile dominant ground contact"
        return "Increase textile dominance on outsole contact surface"
    if category == "fastener":
        if mutation.get("material_aluminum"):
            return "Switch material from steel to aluminum to qualify for lower-duty classification"
    return "Apply alternate construction to explore lower-duty classification"
