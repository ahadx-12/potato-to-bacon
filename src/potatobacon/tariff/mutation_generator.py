"""Generate structured mutation candidates from a product spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
