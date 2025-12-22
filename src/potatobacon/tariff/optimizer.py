from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from potatobacon.law.solver_z3 import PolicyAtom, analyze_scenario
from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.tariff.atom_utils import atom_provenance
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context

from .atoms_hts import DUTY_RATES
from .engine import apply_mutations, compute_duty_result
from .models import NetSavings, TariffFeasibility, TariffScenario
from .normalizer import normalize_compiled_facts


@dataclass
class OptimizationResult:
    baseline_rate: float
    optimized_rate: float
    best_mutation: Optional[Dict[str, Any]]
    baseline_scenario: TariffScenario
    optimized_scenario: TariffScenario
    active_codes_baseline: List[str]
    active_codes_optimized: List[str]
    law_context: str
    tariff_manifest_hash: str
    proof_id: str
    proof_payload_hash: str
    provenance_chain: List[Dict[str, Any]]
    status: str


@dataclass
class _ScenarioEvaluation:
    scenario: TariffScenario
    is_sat: bool
    duty_rate: Optional[float]
    duty_atoms: List[PolicyAtom]
    active_atoms: List[PolicyAtom]
    unsat_core: List[PolicyAtom]
    provenance: List[Dict[str, Any]]
    duty_status: str


def compute_net_savings_projection(
    *,
    baseline_rate: float | None,
    optimized_rate: float | None,
    declared_value_per_unit: float,
    annual_volume: int | None,
    feasibility: TariffFeasibility | None = None,
) -> NetSavings:
    feasibility = feasibility or TariffFeasibility()
    if baseline_rate is None or optimized_rate is None or annual_volume is None:
        return NetSavings()
    gross = (baseline_rate - optimized_rate) / 100.0 * declared_value_per_unit * annual_volume
    implementation_cost = feasibility.one_time_cost + feasibility.recurring_cost_per_unit * annual_volume
    first_year_adjustment = max(0.0, (365 - feasibility.implementation_time_days) / 365)
    first_year_savings = gross * first_year_adjustment
    net = first_year_savings - implementation_cost
    payback_months = None
    if gross > 0:
        monthly = gross / 12.0
        if monthly > 0:
            payback_months = implementation_cost / monthly
    return NetSavings(
        gross_duty_savings=gross,
        first_year_savings=first_year_savings,
        net_annual_savings=net,
        payback_months=payback_months,
    )


def _build_provenance(duty_atoms: List[PolicyAtom], scenario_label: str) -> List[Dict[str, Any]]:
    return [atom_provenance(atom, scenario_label) for atom in duty_atoms]


def _evaluate_scenario(
    atoms: List[PolicyAtom], scenario: TariffScenario, scenario_label: str, duty_rates: Dict[str, float]
) -> _ScenarioEvaluation:
    is_sat, active_atoms, unsat_core = analyze_scenario(scenario.facts, atoms)
    duty_result = compute_duty_result(
        atoms, scenario, active_atoms=active_atoms, is_sat=is_sat, duty_rates=duty_rates
    )
    duty_atoms = duty_result.active_atoms or [atom for atom in active_atoms if atom.source_id in duty_rates]
    duty_rate: Optional[float] = duty_result.duty_rate
    if is_sat and duty_atoms and duty_rate is None:
        duty_rate = float(duty_rates[duty_atoms[-1].source_id])
    provenance = _build_provenance(duty_atoms, scenario_label)
    return _ScenarioEvaluation(
        scenario=scenario,
        is_sat=is_sat,
        duty_rate=duty_rate,
        duty_atoms=duty_atoms,
        active_atoms=active_atoms,
        unsat_core=unsat_core,
        provenance=provenance,
        duty_status=duty_result.status,
    )


def optimize_tariff(
    base_facts: Dict[str, Any],
    candidate_mutations: Dict[str, List[Any]],
    law_context: Optional[str] = None,
    seed: int = 2025,
) -> OptimizationResult:
    resolved_context = law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    context = context_meta["context_id"]
    duty_rates = context_meta.get("duty_rates") or DUTY_RATES

    normalized_base, _ = normalize_compiled_facts(base_facts)
    baseline_scenario = TariffScenario(name="baseline", facts=normalized_base)
    baseline_eval = _evaluate_scenario(atoms, baseline_scenario, "baseline", duty_rates)

    best_rate = baseline_eval.duty_rate
    best_mutation: Optional[Dict[str, Any]] = None
    best_eval = baseline_eval

    for key in sorted(candidate_mutations.keys()):
        for value in candidate_mutations[key]:
            mutated_scenario = apply_mutations(baseline_scenario, {key: value})
            normalized_mutated, _ = normalize_compiled_facts(mutated_scenario.facts)
            mutated_scenario = TariffScenario(name=mutated_scenario.name, facts=normalized_mutated)
            evaluation = _evaluate_scenario(atoms, mutated_scenario, "optimized", duty_rates)
            if evaluation.duty_rate is None:
                continue
            if best_rate is None or evaluation.duty_rate < best_rate:
                best_rate = evaluation.duty_rate
                best_mutation = {key: value}
                best_eval = evaluation

    status: str
    optimized_rate: float
    baseline_rate: float

    if best_rate is None:
        status = "INFEASIBLE"
        optimized_rate = 0.0
        baseline_rate = 0.0 if baseline_eval.duty_rate is None else baseline_eval.duty_rate
    else:
        optimized_rate = best_rate
        baseline_rate = baseline_eval.duty_rate if baseline_eval.duty_rate is not None else optimized_rate
        status = "OPTIMIZED" if best_mutation else "BASELINE"

    provenance_chain: List[Dict[str, Any]] = []
    provenance_chain.extend(baseline_eval.provenance)
    if best_eval is not baseline_eval:
        provenance_chain.extend(best_eval.provenance)
    provenance_chain.sort(
        key=lambda item: (
            item.get("source_id", ""),
            item.get("section", ""),
            item.get("text", ""),
            item.get("scenario", ""),
        )
    )

    proof_handle = record_tariff_proof(
        law_context=context,
        base_facts=normalized_base,
        mutations={"candidates": candidate_mutations, "applied": best_mutation or {}},
        baseline_active=baseline_eval.active_atoms,
        optimized_active=best_eval.active_atoms,
        baseline_sat=baseline_eval.is_sat,
        optimized_sat=best_eval.is_sat,
        baseline_duty_rate=baseline_eval.duty_rate,
        optimized_duty_rate=best_eval.duty_rate,
        baseline_duty_status=baseline_eval.duty_status,
        optimized_duty_status=best_eval.duty_status,
        baseline_scenario=baseline_scenario.facts,
        optimized_scenario=best_eval.scenario.facts,
        baseline_unsat_core=baseline_eval.unsat_core,
        optimized_unsat_core=best_eval.unsat_core,
        provenance_chain=provenance_chain,
        tariff_manifest_hash=context_meta["manifest_hash"],
    )

    return OptimizationResult(
        baseline_rate=baseline_rate,
        optimized_rate=optimized_rate,
        best_mutation=best_mutation,
        baseline_scenario=baseline_scenario,
        optimized_scenario=best_eval.scenario,
        active_codes_baseline=sorted([atom.source_id for atom in baseline_eval.duty_atoms]),
        active_codes_optimized=sorted([atom.source_id for atom in best_eval.duty_atoms]),
        law_context=context,
        tariff_manifest_hash=context_meta["manifest_hash"],
        proof_id=proof_handle.proof_id,
        proof_payload_hash=proof_handle.proof_payload_hash,
        provenance_chain=provenance_chain,
        status=status,
    )
