"""Deterministic compiler that converts ProductSpecModel into solver facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .product_schema import ProductCategory, ProductSpecModel
from .fact_schema_registry import FactSchemaRegistry
from .fact_vocabulary import (
    CATEGORY_TO_CHAPTERS,
    MATERIAL_CHAPTER_TOKENS,
    expand_facts,
)
from .sku_models import SKURecordModel


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


def _origin_fact_key(country: str | None) -> str | None:
    if not country:
        return None
    return f"origin_country_{country.upper()}"


def compile_facts(
    product: ProductSpecModel, bom_signals: Dict[str, Any] | None = None
) -> Tuple[Dict[str, Any], List[DerivedFactEvidence]]:
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
        facts["product_type_chassis_bolt"] = True
        _add_evidence(
            evidence,
            fact_key="product_type_chassis_bolt",
            value=True,
            derived_from_fields=["product_category"],
            calculation="fastener default to chassis bolt",
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

    if product.product_category == ProductCategory.ELECTRONICS:
        facts["product_type_electronics"] = True
        _add_evidence(
            evidence,
            fact_key="product_type_electronics",
            value=True,
            derived_from_fields=["product_category"],
            calculation="electronics category",
        )
    if product.product_category == ProductCategory.APPAREL_TEXTILE:
        facts["product_type_apparel_textile"] = True
        _add_evidence(
            evidence,
            fact_key="product_type_apparel_textile",
            value=True,
            derived_from_fields=["product_category"],
            calculation="apparel/textile category",
        )

    electronics_mapping = {
        "contains_pcb": product.has_pcb,
        "electronics_cable_or_connector": product.is_cable_or_connector,
        "electronics_enclosure": product.is_enclosure_or_housing,
        "contains_battery": product.contains_battery,
    }
    for fact_key, flag_value in electronics_mapping.items():
        facts[fact_key] = bool(flag_value)
        _add_evidence(
            evidence,
            fact_key=fact_key,
            value=bool(flag_value),
            derived_from_fields=[fact_key.replace("electronics_", "").replace("contains_", "has_")],
            calculation="boolean mapping",
        )

    if product.product_category == ProductCategory.ELECTRONICS:
        insulated_hint = facts.get("material_plastic") or facts.get("material_rubber")
        if product.is_cable_or_connector:
            facts["electronics_has_connectors"] = True
            _add_evidence(
                evidence,
                fact_key="electronics_has_connectors",
                value=True,
                derived_from_fields=["is_cable_or_connector"],
                calculation="cable/connector keyword evidence",
            )
            facts["electronics_is_cable_assembly"] = True
            _add_evidence(
                evidence,
                fact_key="electronics_is_cable_assembly",
                value=True,
                derived_from_fields=["is_cable_or_connector"],
                calculation="cable/connector keyword evidence",
            )
            if insulated_hint:
                facts["electronics_insulated_conductors"] = True
                _add_evidence(
                    evidence,
                    fact_key="electronics_insulated_conductors",
                    value=True,
                    derived_from_fields=["materials"],
                    calculation="plastic or rubber material implies insulation",
                )
        elif product.wire_harness_present:
            facts["electronics_has_connectors"] = True
            _add_evidence(
                evidence,
                fact_key="electronics_has_connectors",
                value=True,
                derived_from_fields=["wire_harness_present"],
                calculation="harness indicator implies connectors",
            )
            facts["electronics_is_cable_assembly"] = True
            _add_evidence(
                evidence,
                fact_key="electronics_is_cable_assembly",
                value=True,
                derived_from_fields=["wire_harness_present"],
                calculation="harness indicator implies assembly",
            )
            facts["electronics_insulated_conductors"] = True
            _add_evidence(
                evidence,
                fact_key="electronics_insulated_conductors",
                value=True,
                derived_from_fields=["wire_harness_present"],
                calculation="wire harness implies insulated conductors",
            )
        if product.voltage_rating_known is not None:
            facts["electronics_voltage_rating_known"] = bool(product.voltage_rating_known)
            _add_evidence(
                evidence,
                fact_key="electronics_voltage_rating_known",
                value=bool(product.voltage_rating_known),
                derived_from_fields=["voltage_rating_known"],
                calculation="explicit voltage hint",
            )

    apparel_mapping = {
        "textile_knit": product.is_knit,
        "textile_woven": product.is_woven,
        "has_coating_or_lamination": product.has_coating_or_lamination,
    }
    for fact_key, flag_value in apparel_mapping.items():
        facts[fact_key] = bool(flag_value)
        _add_evidence(
            evidence,
            fact_key=fact_key,
            value=bool(flag_value),
            derived_from_fields=[fact_key.replace("textile_", "is_")],
            calculation="boolean mapping",
        )

    if product.fiber_cotton_pct is not None:
        cotton_dominant = product.fiber_cotton_pct >= 50
        facts["fiber_cotton_dominant"] = cotton_dominant
        _add_evidence(
            evidence,
            fact_key="fiber_cotton_dominant",
            value=cotton_dominant,
            derived_from_fields=["fiber_cotton_pct"],
            calculation=f"cotton_pct={product.fiber_cotton_pct}",
        )
    else:
        facts["fiber_cotton_dominant"] = False
    if product.fiber_polyester_pct is not None:
        polyester_dominant = product.fiber_polyester_pct >= 50
        facts["fiber_polyester_dominant"] = polyester_dominant
        _add_evidence(
            evidence,
            fact_key="fiber_polyester_dominant",
            value=polyester_dominant,
            derived_from_fields=["fiber_polyester_pct"],
            calculation=f"polyester_pct={product.fiber_polyester_pct}",
        )
    else:
        facts["fiber_polyester_dominant"] = False

    if product.fiber_nylon_pct is not None:
        facts["fiber_nylon_present"] = product.fiber_nylon_pct > 0
        _add_evidence(
            evidence,
            fact_key="fiber_nylon_present",
            value=product.fiber_nylon_pct > 0,
            derived_from_fields=["fiber_nylon_pct"],
            calculation=f"nylon_pct={product.fiber_nylon_pct}",
        )

    if bom_signals:
        dominant_material = bom_signals.get("dominant_material")
        if dominant_material:
            fact_key = MATERIAL_FACTS.get(dominant_material.lower())
            if fact_key:
                facts[fact_key] = True
                _add_evidence(
                    evidence,
                    fact_key=fact_key,
                    value=True,
                    derived_from_fields=["bom"],
                    calculation=f"bom dominant material: {dominant_material}",
                )
        primary_origin = bom_signals.get("primary_origin")
        if primary_origin:
            origin_key = _origin_fact_key(primary_origin)
            if origin_key:
                facts[origin_key] = True
                _add_evidence(
                    evidence,
                    fact_key=origin_key,
                    value=True,
                    derived_from_fields=["bom"],
                    calculation="bom primary origin",
                )

    origin_key = _origin_fact_key(product.origin_country)
    if origin_key:
        facts[origin_key] = True
        _add_evidence(
            evidence,
            fact_key=origin_key,
            value=True,
            derived_from_fields=["origin_country"],
            calculation="direct mapping",
        )
    facts["origin_country_raw"] = product.origin_country
    facts["export_country"] = product.export_country
    facts["import_country"] = product.import_country

    if origin_key is None:
        facts["requires_origin_data"] = True
        _add_evidence(
            evidence,
            fact_key="requires_origin_data",
            value=True,
            derived_from_fields=["origin_country"],
            calculation="origin missing",
        )
    else:
        facts["requires_origin_data"] = False

    if product.import_country and product.origin_country:
        if product.import_country.upper() == "US" and product.origin_country.upper() in {"CA", "MX"}:
            facts["fta_usmca_eligible"] = True
            _add_evidence(
                evidence,
                fact_key="fta_usmca_eligible",
                value=True,
                derived_from_fields=["origin_country", "import_country"],
                calculation="US import from USMCA partner",
            )
            facts["duty_reduction_possible"] = True
            _add_evidence(
                evidence,
                fact_key="duty_reduction_possible",
                value=True,
                derived_from_fields=["fta_usmca_eligible"],
                calculation="FTA eligibility indicator",
            )

    if product.product_category == ProductCategory.FASTENER and product.origin_country:
        if product.origin_country.upper() == "CN":
            facts["ad_cvd_possible"] = True
            _add_evidence(
                evidence,
                fact_key="ad_cvd_possible",
                value=True,
                derived_from_fields=["origin_country", "product_category"],
                calculation="CN origin fastener triggers AD/CVD review",
            )
    else:
        facts.setdefault("ad_cvd_possible", False)

    if facts.get("material_steel") or facts.get("material_aluminum"):
        facts.setdefault("material_metal", True)
    else:
        facts.setdefault("material_metal", False)

    # --- Vocabulary bridge: emit chapter/category and synonym tokens ---
    # Add chapter/category tokens based on product category
    for ch_token in CATEGORY_TO_CHAPTERS.get(product.product_category, []):
        facts[ch_token] = True
        _add_evidence(
            evidence,
            fact_key=ch_token,
            value=True,
            derived_from_fields=["product_category"],
            calculation=f"vocabulary bridge: {product.product_category.value} -> {ch_token}",
        )

    # Add material-derived chapter tokens
    for mat_key, ch_tokens in MATERIAL_CHAPTER_TOKENS.items():
        if facts.get(mat_key) is True:
            for ch_tok in ch_tokens:
                if ch_tok not in facts:
                    facts[ch_tok] = True
                    _add_evidence(
                        evidence,
                        fact_key=ch_tok,
                        value=True,
                        derived_from_fields=["materials"],
                        calculation=f"vocabulary bridge: {mat_key} -> {ch_tok}",
                    )

    # Emit synonym tokens for guard compatibility
    if product.product_category == ProductCategory.FASTENER:
        facts["product_type_fastener"] = True
        _add_evidence(
            evidence,
            fact_key="product_type_fastener",
            value=True,
            derived_from_fields=["product_category"],
            calculation="vocabulary bridge: is_fastener synonym",
        )

    if product.product_category == ProductCategory.APPAREL_TEXTILE:
        facts["product_type_apparel"] = True
        _add_evidence(
            evidence,
            fact_key="product_type_apparel",
            value=True,
            derived_from_fields=["product_category"],
            calculation="vocabulary bridge: apparel_textile synonym",
        )

    return facts, evidence


@dataclass(frozen=True)
class CompiledFacts:
    facts: Dict[str, Any]
    provenance: Dict[str, Any]


class FactCompiler:
    def __init__(self, registry: FactSchemaRegistry | None = None) -> None:
        self.registry = registry or FactSchemaRegistry()

    def _evidence_facts(self, sku: SKURecordModel, session: Any) -> Dict[str, Any]:
        evidence_sources: List[Dict[str, Any]] = []
        if isinstance(getattr(session, "evidence_facts", None), dict):
            evidence_sources.append(session.evidence_facts)
        if isinstance(getattr(session, "extracted_facts", None), dict):
            evidence_sources.append(session.extracted_facts)
        metadata = getattr(sku, "metadata", None) or {}
        if isinstance(metadata, dict) and isinstance(metadata.get("evidence_facts"), dict):
            evidence_sources.append(metadata["evidence_facts"])
        merged: Dict[str, Any] = {}
        for source in evidence_sources:
            for key, value in source.items():
                merged.setdefault(key, value)
        return merged

    def _sku_field_mapping(self, sku: SKURecordModel) -> Dict[str, Any]:
        return {
            "origin_country": sku.origin_country,
            "export_country": sku.export_country,
            "import_country": sku.import_country,
            "declared_value_per_unit": sku.declared_value_per_unit,
            "annual_volume": sku.annual_volume,
            "current_hts": sku.current_hts,
            "inferred_category": sku.inferred_category,
        }

    def compile(self, sku: SKURecordModel, session: Any) -> CompiledFacts:
        schema = self.registry.get_all_facts_for_sku(sku)
        overrides = getattr(session, "fact_overrides", None) or {}
        evidence_facts = self._evidence_facts(sku, session)
        sku_fields = self._sku_field_mapping(sku)

        facts: Dict[str, Any] = {}
        provenance: Dict[str, Any] = {}

        for key in sorted(schema.keys()):
            if key in overrides:
                override_val = overrides[key]
                value = getattr(override_val, "value", override_val)
                facts[key] = value
                provenance[key] = {"source": "override"}
            elif key in evidence_facts:
                facts[key] = evidence_facts[key]
                provenance[key] = {"source": "evidence"}
            elif key in sku_fields:
                facts[key] = sku_fields[key]
                provenance[key] = {"source": "sku"}
            else:
                facts[key] = None
                provenance[key] = {"source": "default"}

        return CompiledFacts(facts=facts, provenance=provenance)
