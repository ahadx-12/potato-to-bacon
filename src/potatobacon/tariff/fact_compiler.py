"""Deterministic compiler that converts ProductSpecModel into solver facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .product_schema import ProductCategory, ProductSpecModel


@dataclass
class DerivedFactEvidence:
    """Captures why a derived fact exists for auditability."""

    fact_key: str
    value: Any
    derived_from_fields: List[str]
    calculation: str


MATERIAL_FACTS = {
    "steel": "material_steel",
    "aluminum": "material_aluminum",
    "textile": "material_textile",
    "rubber": "material_rubber",
    "leather": "material_leather",
    "plastic": "material_plastic",
    "synthetic": "material_synthetic",
}

SURFACE_CONTACT_FACTS = {
    "textile": "surface_contact_textile_gt_50",
    "rubber": "surface_contact_rubber_gt_50",
    "leather": "surface_contact_leather_gt_50",
}


def _material_presence(materials: Iterable[str]) -> Dict[str, bool]:
    presence: Dict[str, bool] = {fact: False for fact in MATERIAL_FACTS.values()}
    for material in materials:
        key = MATERIAL_FACTS.get(material.lower())
        if key:
            presence[key] = True
    return presence


def _surface_coverage_flags(coverages: Iterable[Tuple[str, float]]) -> Dict[str, bool]:
    flags: Dict[str, bool] = {fact: False for fact in SURFACE_CONTACT_FACTS.values()}
    for material, percent in coverages:
        if percent is None:
            continue
        fact_key = SURFACE_CONTACT_FACTS.get(material.lower())
        if fact_key and percent >= 50.0:
            flags[fact_key] = True
    return flags


def _add_evidence(
    evidence: List[DerivedFactEvidence],
    fact_key: str,
    value: Any,
    derived_from_fields: List[str],
    calculation: str,
) -> None:
    evidence.append(
        DerivedFactEvidence(
            fact_key=fact_key,
            value=value,
            derived_from_fields=derived_from_fields,
            calculation=calculation,
        )
    )


def compile_facts(product: ProductSpecModel) -> Tuple[Dict[str, Any], List[DerivedFactEvidence]]:
    """
    Compile a :class:`ProductSpecModel` into solver-friendly facts and evidence.

    Returns a tuple of (facts, compiled_evidence).
    """

    facts: Dict[str, Any] = {}
    evidence: List[DerivedFactEvidence] = []

    facts["product_category"] = product.product_category.value
    _add_evidence(
        evidence,
        fact_key="product_category",
        value=product.product_category.value,
        derived_from_fields=["product_category"],
        calculation="direct mapping",
    )

    material_names = [material.material.lower() for material in product.materials]
    material_flags = _material_presence(material_names)
    facts.update(material_flags)
    for material_name in material_names:
        fact_key = MATERIAL_FACTS.get(material_name)
        if fact_key:
            _add_evidence(
                evidence,
                fact_key=fact_key,
                value=True,
                derived_from_fields=["materials"],
                calculation=f"material present: {material_name}",
            )

    coverage_tuples = [
        (coverage.material.lower(), coverage.percent_coverage)
        for coverage in product.surface_coverage
    ]
    surface_flags = _surface_coverage_flags(coverage_tuples)
    facts.update(surface_flags)
    for coverage in product.surface_coverage:
        fact_key = SURFACE_CONTACT_FACTS.get(coverage.material.lower())
        if fact_key and coverage.percent_coverage is not None:
            _add_evidence(
                evidence,
                fact_key=fact_key,
                value=facts[fact_key],
                derived_from_fields=["surface_coverage"],
                calculation=f"coverage {coverage.percent_coverage}%",
            )

    if product.product_category == ProductCategory.FASTENER:
        facts["is_fastener"] = True
        _add_evidence(
            evidence,
            fact_key="is_fastener",
            value=True,
            derived_from_fields=["product_category"],
            calculation="category fastener implies fastener role",
        )

    if product.use_function:
        facts[f"use_function_{product.use_function.lower()}"] = True
        _add_evidence(
            evidence,
            fact_key=f"use_function_{product.use_function.lower()}",
            value=True,
            derived_from_fields=["use_function"],
            calculation="direct mapping",
        )

    if product.manufacturing_process:
        key = f"manufacturing_{product.manufacturing_process.value}"
        facts[key] = True
        _add_evidence(
            evidence,
            fact_key=key,
            value=True,
            derived_from_fields=["manufacturing_process"],
            calculation="direct mapping",
        )

    if product.declared_value_per_unit is not None:
        facts["declared_value_per_unit"] = product.declared_value_per_unit
        _add_evidence(
            evidence,
            fact_key="declared_value_per_unit",
            value=product.declared_value_per_unit,
            derived_from_fields=["declared_value_per_unit"],
            calculation="direct mapping",
        )
    if product.annual_volume is not None:
        facts["annual_volume"] = product.annual_volume
        _add_evidence(
            evidence,
            fact_key="annual_volume",
            value=product.annual_volume,
            derived_from_fields=["annual_volume"],
            calculation="direct mapping",
        )

    return facts, evidence
