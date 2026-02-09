from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from potatobacon.law.arbitrage_hunter import ArbitrageHunter
from potatobacon.law.solver_z3 import PolicyAtom, analyze_scenario, check_scenario
from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.overlays import effective_duty_rate, evaluate_overlays
from potatobacon.tariff.origin_engine import build_origin_policy_atoms

from .atom_utils import atom_provenance
from .atoms_hts import DUTY_RATES
from .models import TariffDossierModel, TariffExplainResponseModel, TariffScenario


@dataclass(frozen=True)
class DutyResult:
    status: str
    duty_rate: float | None
    duty_atom_ids: List[str]
    active_atoms: List[PolicyAtom]
    missing_facts: List[str]


def _active_duty_atoms(
    atoms: Iterable[PolicyAtom],
    scenario: TariffScenario,
    duty_rates: Mapping[str, float],
) -> Tuple[bool, List[PolicyAtom]]:
    """Return active duty-bearing atoms for a scenario."""

    origin_atoms = build_origin_policy_atoms()
    combined_atoms = list(origin_atoms) + list(atoms)
    is_sat, active_atoms = check_scenario(scenario.facts, combined_atoms)
    duty_atoms = [atom for atom in active_atoms if atom.source_id in duty_rates]
    return is_sat, duty_atoms


def compute_duty_result(
    atoms: Iterable[PolicyAtom],
    scenario: TariffScenario,
    *,
    active_atoms: List[PolicyAtom] | None = None,
    is_sat: bool | None = None,
    duty_rates: Mapping[str, float] | None = None,
) -> DutyResult:
    """Compute the duty outcome for ``scenario`` without throwing."""

    rates = duty_rates or DUTY_RATES
    if active_atoms is None or is_sat is None:
        is_sat, duty_atoms = _active_duty_atoms(atoms, scenario, rates)
    else:
        duty_atoms = [atom for atom in active_atoms if atom.source_id in rates]

    if not is_sat:
        return DutyResult(
            status="UNSAT",
            duty_rate=None,
            duty_atom_ids=[],
            active_atoms=duty_atoms,
            missing_facts=[],
        )
    if not duty_atoms:
        return DutyResult(
            status="NO_DUTY_RULE_ACTIVE",
            duty_rate=None,
            duty_atom_ids=[],
            active_atoms=duty_atoms,
            missing_facts=[],
        )
    active_atom = duty_atoms[-1]
    duty_rate = float(rates[active_atom.source_id])
    return DutyResult(
        status="OK",
        duty_rate=duty_rate,
        duty_atom_ids=[atom.source_id for atom in duty_atoms],
        active_atoms=duty_atoms,
        missing_facts=[],
    )


def compute_duty_rate(
    atoms: Iterable[PolicyAtom],
    scenario: TariffScenario,
    *,
    strict: bool = True,
    duty_rates: Mapping[str, float] | None = None,
) -> float | None:
    """Compute the duty rate for ``scenario`` using the provided atoms."""

    result = compute_duty_result(atoms, scenario, duty_rates=duty_rates)
    if result.status == "UNSAT":
        if strict:
            raise ValueError("Scenario is logically inconsistent with the tariff atoms")
        return result.duty_rate
    if result.status != "OK":
        if strict:
            raise ValueError("No duty rule activated for scenario")
        return result.duty_rate
    return result.duty_rate


def apply_mutations(base_scenario: TariffScenario, mutations: Dict[str, Any]) -> TariffScenario:
    """Apply fact mutations with Z3-driven entailment derivations.

    Instead of hardcoded felt logic, this applies a chain of entailment
    rules that derive secondary facts from primary mutations.  Each rule
    is an ``if condition -> set fact`` pair, evaluated until no new facts
    are derived (fixed-point).
    """

    mutated_facts = deepcopy(base_scenario.facts)
    mutated_facts.update(mutations)

    # Entailment rules: (condition_func, derived_facts)
    _entailment_rules = [
        # Felt covering implies textile dominance on ground contact
        (
            lambda f: f.get("felt_covering_gt_50"),
            {"surface_contact_textile_gt_50": True, "surface_contact_rubber_gt_50": False},
        ),
        # Material metal derived from steel or aluminum
        (
            lambda f: f.get("material_steel") or f.get("material_aluminum"),
            {"material_metal": True},
        ),
        # USMCA assembly with qualifying components -> duty-free eligible
        (
            lambda f: f.get("assembled_in_usmca") and f.get("origin_component_qualifying"),
            {"fta_usmca_eligible": True, "duty_reduction_possible": True},
        ),
        # Green energy certification for EV components -> exemption eligible
        (
            lambda f: f.get("green_energy_certified") and f.get("end_use_electric_vehicle"),
            {"green_energy_exemption_eligible": True},
        ),
        # Battery implies electronics category marker
        (
            lambda f: f.get("contains_battery") and f.get("battery_type_lithium_ion"),
            {"product_type_battery": True, "product_type_electronics": True},
        ),
        # Cable assembly entailments
        (
            lambda f: f.get("electronics_is_cable_assembly") and f.get("electronics_has_connectors"),
            {"electronics_cable_or_connector": True},
        ),
    ]

    # Fixed-point evaluation: apply rules until no new facts are derived
    changed = True
    iterations = 0
    while changed and iterations < 10:
        changed = False
        iterations += 1
        for condition, derived in _entailment_rules:
            if condition(mutated_facts):
                for key, value in derived.items():
                    if mutated_facts.get(key) != value:
                        mutated_facts[key] = value
                        changed = True

    return TariffScenario(name=f"{base_scenario.name}-mutated", facts=mutated_facts)


def _build_provenance(duty_atoms: Iterable[PolicyAtom], scenario_label: str) -> List[Dict[str, Any]]:
    provenance: list[Dict[str, Any]] = []
    for atom in duty_atoms:
        entry = atom_provenance(atom, scenario_label)
        provenance.append(entry)
    return provenance


def run_tariff_hack(
    base_facts: Dict[str, Any],
    mutations: Dict[str, Any] | None = None,
    law_context: str | None = None,
    seed: int = 2025,
    evidence_pack: Dict[str, Any] | None = None,
) -> TariffDossierModel:
    """Compute baseline vs optimized tariff outcomes for a given scenario."""

    resolved_context = law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    context = context_meta["context_id"]
    ArbitrageHunter(atoms, seed=seed)  # placeholder wiring for future GA usage

    duty_rates = context_meta.get("duty_rates") or DUTY_RATES
    baseline = TariffScenario(name="baseline", facts=deepcopy(base_facts))
    baseline_result = compute_duty_result(atoms, baseline, duty_rates=duty_rates)
    baseline_overlays = evaluate_overlays(
        facts=baseline.facts,
        active_codes=[atom.source_id for atom in baseline_result.active_atoms],
        origin_country=base_facts.get("origin_country"),
        import_country=base_facts.get("import_country"),
        hts_code=base_facts.get("hts_code"),
    )
    baseline_rate_raw = baseline_result.duty_rate if baseline_result.duty_rate is not None else 0.0
    baseline_effective = effective_duty_rate(baseline_rate_raw, baseline_overlays)

    if mutations:
        optimized = apply_mutations(baseline, mutations)
    else:
        optimized = TariffScenario(name="baseline", facts=deepcopy(base_facts))
    optimized_result = compute_duty_result(atoms, optimized, duty_rates=duty_rates)
    optimized_overlays = evaluate_overlays(
        facts=optimized.facts,
        active_codes=[atom.source_id for atom in optimized_result.active_atoms],
        origin_country=optimized.facts.get("origin_country_raw") or base_facts.get("origin_country"),
        import_country=optimized.facts.get("import_country"),
        hts_code=optimized.facts.get("hts_code"),
    )
    optimized_rate_raw = optimized_result.duty_rate if optimized_result.duty_rate is not None else baseline_rate_raw
    optimized_effective = effective_duty_rate(optimized_rate_raw, optimized_overlays)

    stop_optimization_flag = any(ov.stop_optimization for ov in baseline_overlays + optimized_overlays)
    status = "REQUIRES_REVIEW" if stop_optimization_flag else ("OPTIMIZED" if optimized_effective < baseline_effective else "BASELINE")
    savings = baseline_effective - optimized_effective

    sat_baseline, active_atoms_baseline, unsat_core_baseline = analyze_scenario(
        baseline.facts, list(build_origin_policy_atoms()) + list(atoms)
    )
    sat_optimized, active_atoms_optimized, unsat_core_optimized = analyze_scenario(
        optimized.facts, list(build_origin_policy_atoms()) + list(atoms)
    )
    duty_atoms_baseline = [atom for atom in active_atoms_baseline if atom.source_id in duty_rates]
    duty_atoms_optimized = [atom for atom in active_atoms_optimized if atom.source_id in duty_rates]

    provenance_chain: list[Dict[str, Any]] = []
    provenance_chain.extend(_build_provenance(duty_atoms_baseline, "baseline"))
    provenance_chain.extend(_build_provenance(duty_atoms_optimized, "optimized"))
    provenance_chain.sort(
        key=lambda item: (
            item.get("source_id", ""),
            item.get("section", ""),
            item.get("text", ""),
            item.get("scenario", ""),
        )
    )

    metrics = {
        "status": status,
        "seed": seed,
        "sat_baseline": sat_baseline,
        "sat_optimized": sat_optimized,
        "law_context": context,
        "tariff_manifest_hash": context_meta["manifest_hash"],
        "baseline_duty_status": baseline_result.status,
        "optimized_duty_status": optimized_result.status,
        "baseline_overlays": len(baseline_overlays),
        "optimized_overlays": len(optimized_overlays),
        "overlay_stop_optimization": stop_optimization_flag,
        "baseline_effective_duty_rate": baseline_effective,
        "optimized_effective_duty_rate": optimized_effective,
    }

    proof_handle = record_tariff_proof(
        law_context=context,
        base_facts=baseline.facts,
        mutations=mutations,
        baseline_active=active_atoms_baseline,
        optimized_active=active_atoms_optimized,
        baseline_sat=sat_baseline,
        optimized_sat=sat_optimized,
        baseline_duty_rate=baseline_rate_raw,
        optimized_duty_rate=optimized_rate_raw,
        baseline_duty_status=baseline_result.status,
        optimized_duty_status=optimized_result.status,
        baseline_scenario=baseline.facts,
        optimized_scenario=optimized.facts,
        baseline_unsat_core=unsat_core_baseline,
        optimized_unsat_core=unsat_core_optimized,
        provenance_chain=provenance_chain,
        evidence_pack=evidence_pack,
        overlays={
            "baseline": [item.model_dump() for item in baseline_overlays],
            "optimized": [item.model_dump() for item in optimized_overlays],
        },
        tariff_manifest_hash=context_meta["manifest_hash"],
    )

    dossier = TariffDossierModel(
        proof_id=proof_handle.proof_id,
        proof_payload_hash=proof_handle.proof_payload_hash,
        law_context=context,
        status=status,
        baseline_duty_rate=baseline_rate_raw,
        optimized_duty_rate=optimized_rate_raw,
        baseline_effective_duty_rate=baseline_effective,
        optimized_effective_duty_rate=optimized_effective,
        savings_per_unit=savings,
        baseline_scenario=baseline.facts,
        optimized_scenario=optimized.facts,
        active_codes_baseline=sorted([atom.source_id for atom in duty_atoms_baseline]),
        active_codes_optimized=sorted([atom.source_id for atom in duty_atoms_optimized]),
        provenance_chain=provenance_chain,
        overlays={
            "baseline": baseline_overlays,
            "optimized": optimized_overlays,
        },
        tariff_manifest_hash=context_meta["manifest_hash"],
        metrics=metrics,
    )
    return dossier


def explain_tariff_scenario(
    base_facts: Dict[str, Any],
    mutations: Dict[str, Any] | None = None,
    law_context: str | None = None,
) -> TariffExplainResponseModel:
    """Return SAT/UNSAT explanation for a tariff scenario."""

    resolved_context = law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    context = context_meta["context_id"]

    scenario = TariffScenario(name="baseline", facts=deepcopy(base_facts))
    if mutations:
        scenario = apply_mutations(scenario, mutations)

    is_sat, active_atoms, unsat_core = analyze_scenario(scenario.facts, atoms)
    duty_rates = context_meta.get("duty_rates") or DUTY_RATES

    if is_sat:
        explanation = "Scenario is logically consistent with tariff rules."
    else:
        parts = [
            "Conflict detected between tariff conditions:"
        ]
        for atom in unsat_core:
            parts.append(
                f"- {atom.source_id} ({getattr(atom, 'section', '')}) {getattr(atom, 'text', '').strip()}"
            )
        explanation = "\n".join(parts)

    duty_result = compute_duty_result(
        atoms, scenario, active_atoms=active_atoms, is_sat=is_sat, duty_rates=duty_rates
    )
    duty_rate: float | None = duty_result.duty_rate

    proof_handle = record_tariff_proof(
        law_context=context,
        base_facts=scenario.facts,
        mutations=mutations,
        baseline_active=active_atoms,
        optimized_active=active_atoms,
        baseline_sat=is_sat,
        optimized_sat=is_sat,
        baseline_duty_rate=duty_rate,
        optimized_duty_rate=duty_rate,
        baseline_duty_status=duty_result.status,
        optimized_duty_status=duty_result.status,
        baseline_scenario=scenario.facts,
        optimized_scenario=scenario.facts,
        baseline_unsat_core=unsat_core,
        optimized_unsat_core=unsat_core,
        tariff_manifest_hash=context_meta["manifest_hash"],
    )

    compliance_flags = {
        "requires_origin_data": bool(scenario.facts.get("requires_origin_data")),
        "ad_cvd_possible": bool(scenario.facts.get("ad_cvd_possible")),
        "requires_ruling_review": bool(unsat_core),
    }
    recommended_next_inputs: list[str] = []
    if compliance_flags["requires_origin_data"]:
        recommended_next_inputs.append("Provide country of origin by component")
    if not scenario.facts.get("fiber_cotton_dominant") and not scenario.facts.get(
        "fiber_polyester_dominant"
    ):
        recommended_next_inputs.append("Provide fiber percentages")

    return TariffExplainResponseModel(
        status="SAT" if is_sat else "UNSAT",
        explanation=explanation,
        proof_id=proof_handle.proof_id,
        proof_payload_hash=proof_handle.proof_payload_hash,
        law_context=context,
        unsat_core=[
            {
                "source_id": atom.source_id,
                "statute": getattr(atom, "statute", ""),
                "section": getattr(atom, "section", ""),
                "text": getattr(atom, "text", ""),
            }
            for atom in unsat_core
        ],
        compliance_flags=compliance_flags,
        recommended_next_inputs=sorted(set(recommended_next_inputs)),
    )
