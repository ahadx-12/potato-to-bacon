from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

from potatobacon.law.arbitrage_hunter import ArbitrageHunter
from potatobacon.law.solver_z3 import PolicyAtom, analyze_scenario, check_scenario
from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context

from .atoms_hts import DUTY_RATES
from .models import TariffDossierModel, TariffExplainResponseModel, TariffScenario


def _active_duty_atoms(atoms: Iterable[PolicyAtom], scenario: TariffScenario) -> Tuple[bool, List[PolicyAtom]]:
    """Return active duty-bearing atoms for a scenario."""

    is_sat, active_atoms = check_scenario(scenario.facts, list(atoms))
    duty_atoms = [atom for atom in active_atoms if atom.source_id in DUTY_RATES]
    return is_sat, duty_atoms


def compute_duty_rate(atoms: Iterable[PolicyAtom], scenario: TariffScenario) -> float:
    """Compute the duty rate for ``scenario`` using the provided atoms."""

    is_sat, duty_atoms = _active_duty_atoms(atoms, scenario)
    if not is_sat:
        raise ValueError("Scenario is logically inconsistent with the tariff atoms")
    if not duty_atoms:
        raise ValueError("No duty rule activated for scenario")
    active_atom = duty_atoms[-1]
    return float(DUTY_RATES[active_atom.source_id])


def apply_mutations(base_scenario: TariffScenario, mutations: Dict[str, Any]) -> TariffScenario:
    """Apply fact mutations and derived felt logic to a tariff scenario."""

    mutated_facts = deepcopy(base_scenario.facts)
    mutated_facts.update(mutations)

    if mutated_facts.get("felt_covering_gt_50"):
        mutated_facts["surface_contact_textile_gt_50"] = True
        mutated_facts["surface_contact_rubber_gt_50"] = False
    return TariffScenario(name=f"{base_scenario.name}-mutated", facts=mutated_facts)


def _build_provenance(duty_atoms: Iterable[PolicyAtom], scenario_label: str) -> List[Dict[str, Any]]:
    provenance: list[Dict[str, Any]] = []
    for atom in duty_atoms:
        provenance.append(
            {
                "scenario": scenario_label,
                "source_id": atom.source_id,
                "statute": getattr(atom, "statute", ""),
                "section": getattr(atom, "section", ""),
                "text": getattr(atom, "text", ""),
                "jurisdiction": atom.outcome.get("jurisdiction", ""),
            }
        )
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

    baseline = TariffScenario(name="baseline", facts=deepcopy(base_facts))
    baseline_rate = compute_duty_rate(atoms, baseline)

    if mutations:
        optimized = apply_mutations(baseline, mutations)
    else:
        optimized = TariffScenario(name="baseline", facts=deepcopy(base_facts))
    optimized_rate = compute_duty_rate(atoms, optimized)

    status = "OPTIMIZED" if optimized_rate < baseline_rate else "BASELINE"
    savings = baseline_rate - optimized_rate

    sat_baseline, active_atoms_baseline, unsat_core_baseline = analyze_scenario(
        baseline.facts, atoms
    )
    sat_optimized, active_atoms_optimized, unsat_core_optimized = analyze_scenario(
        optimized.facts, atoms
    )
    duty_atoms_baseline = [atom for atom in active_atoms_baseline if atom.source_id in DUTY_RATES]
    duty_atoms_optimized = [atom for atom in active_atoms_optimized if atom.source_id in DUTY_RATES]

    provenance_chain: list[Dict[str, Any]] = []
    provenance_chain.extend(_build_provenance(duty_atoms_baseline, "baseline"))
    provenance_chain.extend(_build_provenance(duty_atoms_optimized, "optimized"))

    metrics = {
        "status": status,
        "seed": seed,
        "sat_baseline": sat_baseline,
        "sat_optimized": sat_optimized,
        "law_context": context,
        "tariff_manifest_hash": context_meta["manifest_hash"],
    }

    proof_id = record_tariff_proof(
        law_context=context,
        base_facts=baseline.facts,
        mutations=mutations,
        baseline_active=active_atoms_baseline,
        optimized_active=active_atoms_optimized,
        baseline_sat=sat_baseline,
        optimized_sat=sat_optimized,
        baseline_unsat_core=unsat_core_baseline,
        optimized_unsat_core=unsat_core_optimized,
        evidence_pack=evidence_pack,
        tariff_manifest_hash=context_meta["manifest_hash"],
    )

    dossier = TariffDossierModel(
        proof_id=proof_id,
        law_context=context,
        status=status,
        baseline_duty_rate=baseline_rate,
        optimized_duty_rate=optimized_rate,
        savings_per_unit=savings,
        baseline_scenario=baseline.facts,
        optimized_scenario=optimized.facts,
        active_codes_baseline=[atom.source_id for atom in duty_atoms_baseline],
        active_codes_optimized=[atom.source_id for atom in duty_atoms_optimized],
        provenance_chain=provenance_chain,
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

    proof_id = record_tariff_proof(
        law_context=context,
        base_facts=scenario.facts,
        mutations=mutations,
        baseline_active=active_atoms,
        optimized_active=active_atoms,
        baseline_sat=is_sat,
        optimized_sat=is_sat,
        baseline_unsat_core=unsat_core,
        optimized_unsat_core=unsat_core,
        tariff_manifest_hash=context_meta["manifest_hash"],
    )

    return TariffExplainResponseModel(
        status="SAT" if is_sat else "UNSAT",
        explanation=explanation,
        proof_id=proof_id,
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
    )
