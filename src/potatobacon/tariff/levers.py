"""Lever library defining defensible tariff optimization pathways.

Each lever bundles a deterministic mutation, feasibility metadata, evidence
requirements, and guardrails that constrain when the lever should be applied.
The goal is to surface auditable, reproducible optimization moves rather than
open-ended heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.fact_requirements import FactRequirementRegistry
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


def _guard_expectations(atom: PolicyAtom) -> Dict[str, bool]:
    expectations: Dict[str, bool] = {}
    for literal in atom.guard:
        negated = literal.startswith("¬")
        fact_key = literal[1:] if negated else literal
        expectations[fact_key] = not negated
    return expectations


def diff_candidate_requirements(
    baseline_candidate: PolicyAtom,
    target_candidate: PolicyAtom,
    *,
    requirement_registry: FactRequirementRegistry | None = None,
) -> List[str]:
    """Return fact keys where the target candidate requires a different state."""

    registry = requirement_registry or FactRequirementRegistry()
    baseline_reqs = _guard_expectations(baseline_candidate)
    target_reqs = _guard_expectations(target_candidate)

    diffs: list[str] = []
    for key, expected in target_reqs.items():
        baseline_expected = baseline_reqs.get(key)
        if baseline_expected is None or baseline_expected != expected:
            registry.describe(key)  # normalize/validate via registry
            diffs.append(key)

    deduped: list[str] = []
    for key in sorted(diffs):
        if key not in deduped:
            deduped.append(key)
    return deduped


def _dimension_for_key(fact_key: str) -> str | None:
    if fact_key.startswith("material_"):
        return "material"
    if fact_key.startswith("surface_contact_") or fact_key.startswith("felt_covering_"):
        return "composition"
    if fact_key.startswith("fiber_") or fact_key.startswith("upper_material_") or fact_key.startswith(
        "outer_sole_material_"
    ):
        return "composition"
    if "assembly" in fact_key or fact_key.startswith("product_type_assembly"):
        return "assembly"
    return None


def _dimension_for_diff(diff_keys: Iterable[str]) -> str | None:
    dimension: str | None = None
    for key in diff_keys:
        key_dimension = _dimension_for_key(key)
        if key_dimension is None:
            return None
        if dimension is None:
            dimension = key_dimension
        elif dimension != key_dimension:
            return None
    return dimension


def _mutation_from_diff(
    *,
    diff_keys: Sequence[str],
    target_guard: Mapping[str, bool],
    baseline_guard: Mapping[str, bool],
    facts: Mapping[str, object],
) -> Dict[str, object]:
    mutation: Dict[str, object] = {}
    dimensions: Dict[str, set[str]] = {}

    for key in diff_keys:
        dimension = _dimension_for_key(key)
        if dimension:
            dimensions.setdefault(dimension, set()).add(key)

        if key in target_guard:
            mutation[key] = target_guard[key]

    for base_key, base_expected in baseline_guard.items():
        dimension = _dimension_for_key(base_key)
        if dimension and dimension in dimensions and base_expected and base_key not in mutation:
            mutation[base_key] = False

    for fact_key, value in facts.items():
        if not isinstance(value, bool):
            continue
        dimension = _dimension_for_key(fact_key)
        if dimension and dimension in dimensions and value and fact_key not in mutation:
            mutation[fact_key] = False

    return mutation


def _evidence_for_keys(keys: Iterable[str], requirement_registry: FactRequirementRegistry) -> List[str]:
    evidence: set[str] = set()
    for key in keys:
        requirement = requirement_registry.describe(key)
        evidence.update(requirement.evidence_types)
    return sorted(evidence)


def _feasibility_defaults() -> Mapping[str, object]:
    return {
        "one_time_cost": 750.0,
        "recurring_cost_per_unit": 0.0,
        "implementation_time_days": 21,
        "requires_recertification": False,
        "supply_chain_risk": "MED",
    }


def _rationale_from_diff(dimension: str, target_candidate: PolicyAtom, diff_keys: Sequence[str]) -> str:
    target_label = getattr(target_candidate, "section", None) or target_candidate.source_id
    fact_list = ", ".join(diff_keys)
    if dimension == "material":
        return f"Shift material stack ({fact_list}) to qualify for lower-duty lane {target_label}."
    if dimension == "composition":
        return f"Adjust composition threshold ({fact_list}) to activate textile-dominant classification {target_label}."
    if dimension == "assembly":
        return f"Alter assembly configuration ({fact_list}) to align with {target_label} requirements."
    return f"Adjust {fact_list} to unlock lower-duty candidate {target_label}."


def generate_candidate_levers(
    *,
    baseline_atom: PolicyAtom,
    atoms: Sequence[PolicyAtom],
    duty_rates: Mapping[str, float],
    facts: Mapping[str, object],
    requirement_registry: FactRequirementRegistry | None = None,
    baseline_rate: float | None = None,
) -> List[LeverModel]:
    """Generate physical levers by contrasting cheaper candidates against the baseline."""

    registry = requirement_registry or FactRequirementRegistry()
    base_guard = _guard_expectations(baseline_atom)
    base_rate = baseline_rate if baseline_rate is not None else duty_rates.get(baseline_atom.source_id)
    if base_rate is None:
        return []

    generated: list[LeverModel] = []
    seen: set[str] = set()
    for candidate in atoms:
        if candidate.source_id == baseline_atom.source_id:
            continue
        candidate_rate = duty_rates.get(candidate.source_id)
        if candidate_rate is None or candidate_rate >= base_rate:
            continue

        diff_keys = diff_candidate_requirements(baseline_atom, candidate, requirement_registry=registry)
        dimension = _dimension_for_diff(diff_keys)
        if not diff_keys or dimension is None:
            continue

        mutation = _mutation_from_diff(
            diff_keys=diff_keys,
            target_guard=_guard_expectations(candidate),
            baseline_guard=base_guard,
            facts=facts,
        )
        if not mutation:
            continue

        lever_id = f"DYNAMIC_{dimension.upper()}::{candidate.source_id}"
        if lever_id in seen:
            continue

        generated.append(
            LeverModel(
                lever_id=lever_id,
                category_scope=["all"],
                required_facts={},
                mutation=mutation,
                rationale=_rationale_from_diff(dimension, candidate, diff_keys),
                feasibility="MED",
                evidence_requirements=_evidence_for_keys(diff_keys, registry),
                feasibility_profile=_feasibility_defaults(),
                risk_floor=25,
            )
        )
        seen.add(lever_id)

    generated.sort(key=lambda lever: lever.lever_id)
    return generated
