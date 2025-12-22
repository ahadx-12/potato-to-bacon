"""Lever library defining defensible tariff optimization pathways.

Each lever bundles a deterministic mutation, feasibility metadata, evidence
requirements, and guardrails that constrain when the lever should be applied.
The goal is to surface auditable, reproducible optimization moves rather than
open-ended heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Sequence

from potatobacon.tariff.product_schema import ProductCategory, ProductSpecModel

Constraint = Callable[[Mapping[str, object]], bool]


@dataclass(frozen=True)
class LeverModel:
    """A single deterministic lever with applicability constraints."""

    lever_id: str
    category_scope: Sequence[str]
    required_facts: Mapping[str, bool | None]
    mutation: Mapping[str, object]
    rationale: str
    feasibility: str
    evidence_requirements: Sequence[str]
    feasibility_profile: Mapping[str, object] | None = field(default=None)
    risk_floor: int = 0
    constraints: Sequence[Constraint] = field(default_factory=list)

    def applies_to(self, *, category: str, facts: Mapping[str, object]) -> bool:
        if "all" not in self.category_scope and category not in self.category_scope:
            return False

        for key, expected in self.required_facts.items():
            if expected is None:
                if key not in facts:
                    return False
                continue
            if facts.get(key) is None:
                return False
            if bool(facts.get(key)) != bool(expected):
                return False

        for constraint in self.constraints:
            if not constraint(facts):
                return False
        return True


def _has_material_evidence(facts: Mapping[str, object]) -> bool:
    return bool(facts.get("material_plastic") or facts.get("material_steel") or facts.get("material_aluminum"))


def _blend_near_threshold(facts: Mapping[str, object]) -> bool:
    cotton = facts.get("fiber_cotton_dominant")
    polyester = facts.get("fiber_polyester_dominant")
    has_both = "fiber_cotton_dominant" in facts and "fiber_polyester_dominant" in facts
    return has_both and (cotton or polyester)


def _requires_knit_confirmation(facts: Mapping[str, object]) -> bool:
    return not bool(facts.get("textile_knit") or facts.get("textile_woven"))


def _no_knit_flip(facts: Mapping[str, object]) -> bool:
    return not (facts.get("textile_knit") and facts.get("textile_woven"))


def _has_insulation_documentation(facts: Mapping[str, object]) -> bool:
    return bool(facts.get("electronics_insulation_documented") or facts.get("electronics_insulated_conductors"))


def _electronics_levers() -> List[LeverModel]:
    return [
        LeverModel(
            lever_id="ELEC_ENCLOSURE_PLASTIC_DOMINANCE",
            category_scope=[ProductCategory.ELECTRONICS.value],
            required_facts={"product_type_electronics": True, "electronics_enclosure": True, "material_steel": True},
            mutation={"material_plastic": True, "material_steel": False},
            rationale="Shift enclosure dominance toward plastics to activate lower-duty enclosure pathways.",
            feasibility="MED",
            evidence_requirements=["Bill of materials showing plastic housing share", "Material declaration from supplier"],
            risk_floor=35,
            constraints=[_has_material_evidence],
        ),
        LeverModel(
            lever_id="ELEC_CONNECTOR_PATHWAY",
            category_scope=[ProductCategory.ELECTRONICS.value],
            required_facts={"product_type_electronics": True, "electronics_cable_or_connector": True},
            mutation={"electronics_cable_or_connector": True, "product_type_electronics": False},
            rationale="Classify device as a cable/connector assembly when harness evidence is present.",
            feasibility="HIGH",
            evidence_requirements=["Connector or harness line items", "Mechanical drawings highlighting connectors"],
            risk_floor=25,
            constraints=[],
        ),
        LeverModel(
            lever_id="ELECTRONICS_CABLE_ASSEMBLY_PATHWAY",
            category_scope=[ProductCategory.ELECTRONICS.value],
            required_facts={
                "product_type_electronics": True,
                "electronics_cable_or_connector": True,
                "electronics_has_connectors": True,
            },
            mutation={
                "electronics_is_cable_assembly": True,
                "electronics_insulated_conductors": True,
                "electronics_voltage_rating_known": True,
            },
            rationale=(
                "Document connectors, insulation, and low-voltage rating to classify as a cable assembly under the "
                "demo electronics lanes."
            ),
            feasibility="HIGH",
            evidence_requirements=[
                "Harness or cable assembly drawing",
                "Connector spec showing voltage/current rating",
                "Jacket or insulation material declaration",
            ],
            risk_floor=25,
            constraints=[],
        ),
        LeverModel(
            lever_id="ELECTRONICS_INSULATION_DOCUMENTATION",
            category_scope=[ProductCategory.ELECTRONICS.value],
            required_facts={
                "product_type_electronics": True,
                "electronics_cable_or_connector": True,
                "electronics_insulated_conductors": None,
            },
            mutation={"electronics_insulated_conductors": True, "electronics_insulation_documented": True},
            rationale="Document insulation on conductors with spec sheet or lab proof to unlock low-voltage lanes.",
            feasibility="HIGH",
            feasibility_profile={
                "one_time_cost": 200.0,
                "recurring_cost_per_unit": 0.0,
                "implementation_time_days": 7,
                "requires_recertification": False,
                "supply_chain_risk": "LOW",
            },
            evidence_requirements=[
                "spec_sheet",
                "manufacturer_datasheet_pdf",
                "lab_test_report",
                "product_photo_label",
            ],
            risk_floor=10,
            constraints=[_has_insulation_documentation],
        ),
        LeverModel(
            lever_id="ELEC_MODULE_PACKAGING",
            category_scope=[ProductCategory.ELECTRONICS.value],
            required_facts={"product_type_electronics": True, "electronics_enclosure": True},
            mutation={"electronics_enclosure": True, "contains_battery": False},
            rationale="Ship as module/housing without active cells to reduce functional classification burden.",
            feasibility="MED",
            evidence_requirements=["Packaging BOM showing enclosure-only configuration", "Removal of active battery cells"],
            risk_floor=40,
            constraints=[],
        ),
    ]


def _apparel_levers() -> List[LeverModel]:
    return [
        LeverModel(
            lever_id="APPAREL_BLEND_DOMINANCE",
            category_scope=[ProductCategory.APPAREL_TEXTILE.value],
            required_facts={"product_type_apparel_textile": True, "fiber_cotton_dominant": True},
            mutation={"fiber_polyester_dominant": True, "fiber_cotton_dominant": False},
            rationale="Document polyester dominance when blend is near 50/50 to access lower synthetic brackets.",
            feasibility="MED",
            evidence_requirements=["Fiber composition test around 50/50 blend", "Lab certificate for polyester share"],
            risk_floor=30,
            constraints=[_blend_near_threshold],
        ),
        LeverModel(
            lever_id="APPAREL_COATING_EXPLICITNESS",
            category_scope=[ProductCategory.APPAREL_TEXTILE.value],
            required_facts={"product_type_apparel_textile": True},
            mutation={"has_coating_or_lamination": True},
            rationale="Make coating/lamination explicit to avoid conservative classification penalties.",
            feasibility="LOW",
            evidence_requirements=["Coating specification", "Supplier declaration of lamination stack"],
            risk_floor=45,
            constraints=[],
        ),
        LeverModel(
            lever_id="APPAREL_CONFIRM_KNIT_WOVEN",
            category_scope=[ProductCategory.APPAREL_TEXTILE.value],
            required_facts={"product_type_apparel_textile": True},
            mutation={"textile_knit": True, "textile_woven": False},
            rationale="Request knit confirmation rather than assuming woven when construction is ambiguous.",
            feasibility="HIGH",
            evidence_requirements=["Construction declaration", "Swatch or lab fabric analysis"],
            risk_floor=20,
            constraints=[_requires_knit_confirmation, _no_knit_flip],
        ),
    ]


def _footwear_levers() -> List[LeverModel]:
    return [
        LeverModel(
            lever_id="FOOTWEAR_SURFACE_TEXTILE_DOMINANCE",
            category_scope=[ProductCategory.FOOTWEAR.value],
            required_facts={"surface_contact_rubber_gt_50": True},
            mutation={"surface_contact_textile_gt_50": True, "surface_contact_rubber_gt_50": False},
            rationale="Move outsole contact above 50% textile using overlays or felt to unlock textile-dominant code.",
            feasibility="MED",
            evidence_requirements=["Outsole coverage measurement", "Material overlay plan"],
            risk_floor=25,
            constraints=[],
        ),
        LeverModel(
            lever_id="FOOTWEAR_FELT_OVERLAY",
            category_scope=[ProductCategory.FOOTWEAR.value],
            required_facts={"surface_contact_rubber_gt_50": True},
            mutation={"felt_covering_gt_50": True, "surface_contact_textile_gt_50": True, "surface_contact_rubber_gt_50": False},
            rationale="Add felt overlay above 50% to treat outsole as textile per GRI Note 4.",
            feasibility="HIGH",
            evidence_requirements=["Material change request", "Coverage diagram"],
            risk_floor=30,
            constraints=[],
        ),
    ]


def _fastener_levers() -> List[LeverModel]:
    return [
        LeverModel(
            lever_id="FASTENER_MATERIAL_SHIFT",
            category_scope=[ProductCategory.FASTENER.value],
            required_facts={"product_type_chassis_bolt": True, "material_steel": True},
            mutation={"material_aluminum": True, "material_steel": False},
            rationale="Shift from steel to aluminum alloy where feasible to capture lower duty lane.",
            feasibility="MED",
            evidence_requirements=["Mechanical strength analysis", "Alloy certification"],
            risk_floor=35,
            constraints=[],
        )
    ]


def lever_library() -> List[LeverModel]:
    """Return the deterministic lever catalog."""

    levers: List[LeverModel] = []
    levers.extend(_electronics_levers())
    levers.extend(_apparel_levers())
    levers.extend(_footwear_levers())
    levers.extend(_fastener_levers())
    return levers


def applicable_levers(*, spec: ProductSpecModel, facts: Mapping[str, object]) -> List[LeverModel]:
    """Filter levers that can safely apply to the compiled facts."""

    category = spec.product_category.value if spec.product_category else "other"
    candidates: List[LeverModel] = []
    for lever in lever_library():
        if lever.applies_to(category=category, facts=facts):
            candidates.append(lever)
    candidates.sort(key=lambda lever: lever.lever_id)
    return candidates
