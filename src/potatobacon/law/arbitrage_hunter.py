"""Arbitrage hunter built on top of the CALE-LAW pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from z3 import BoolVal, If, Optimize, sat  # type: ignore[import-not-found]

from potatobacon.law.cale_metrics import ScenarioMetrics, compute_scenario_metrics, sample_scenarios
from potatobacon.law.solver_z3 import PolicyAtom, build_policy_atoms_from_rules, compile_atoms_to_z3


@dataclass(slots=True)
class ArbitrageRequest:
    jurisdictions: List[str]
    domain: str
    objective: str
    constraints: Dict[str, Any]
    risk_tolerance: str


@dataclass(slots=True)
class ArbitrageCandidate:
    scenario: Dict[str, bool]
    metrics: ScenarioMetrics
    proof_trace: List[str]


@dataclass(slots=True)
class ArbitrageDossier:
    golden_scenario: Dict[str, bool]
    metrics: Dict[str, float]
    proof_trace: List[str]
    risk_flags: List[str]
    candidates: List[ArbitrageCandidate]


class ArbitrageHunter:
    """Explore the CALE rule space for high-value ambiguous scenarios."""

    def __init__(self, atoms: Sequence[PolicyAtom]):
        self._atoms = list(atoms)

    @classmethod
    def from_rules(cls, rules: Sequence[Any]) -> "ArbitrageHunter":
        atoms = build_policy_atoms_from_rules(rules, mapper=None)
        return cls(atoms)

    def _filter_atoms(self, jurisdictions: Sequence[str]) -> List[PolicyAtom]:
        if not jurisdictions:
            return list(self._atoms)
        jurisdictions_lower = {j.lower() for j in jurisdictions}
        return [
            atom
            for atom in self._atoms
            if atom.outcome.get("jurisdiction", "").lower() in jurisdictions_lower
        ]

    def _optimise_seed(self, atoms: Sequence[PolicyAtom], constraints: Mapping[str, Any]) -> Dict[str, bool]:
        optimizer = Optimize()
        var_map = compile_atoms_to_z3(atoms, {})

        # Apply user constraints
        for key, value in constraints.items():
            if key not in var_map:
                continue
            optimizer.add(var_map[key] == BoolVal(bool(value)))

        # Encourage satisfying as many outcomes as possible to expose ambiguity
        objective_terms = []
        for atom in atoms:
            if atom.z3_guard is None or atom.z3_outcome is None:
                continue
            objective_terms.append(If(atom.z3_guard, If(atom.z3_outcome, 1, 1), 0))
        if objective_terms:
            optimizer.maximize(sum(objective_terms))

        result = optimizer.check()
        if result != sat:
            return {}
        model = optimizer.model()
        scenario: Dict[str, bool] = {}
        for name, var in var_map.items():
            value = model.eval(var, model_completion=True)
            scenario[name] = bool(value)
        return scenario

    def _mutate(self, scenario: Dict[str, bool], mutations: int = 2) -> Dict[str, bool]:
        mutated = dict(scenario)
        keys = list(mutated.keys())
        for _ in range(max(1, mutations)):
            key = random.choice(keys)
            mutated[key] = not mutated[key]
        return mutated

    def hunt(self, request: ArbitrageRequest) -> ArbitrageDossier:
        atoms = self._filter_atoms(request.jurisdictions)
        if not atoms:
            atoms = list(self._atoms)

        base_constraints = request.constraints or {}
        seed_scenario = self._optimise_seed(atoms, base_constraints)
        if not seed_scenario:
            seed_candidates = sample_scenarios(atoms, sample_size=5)
        else:
            seed_candidates = [seed_scenario]

        fuzz_budget = 5 if request.risk_tolerance == "low" else 10
        candidates: List[ArbitrageCandidate] = []
        for seed in seed_candidates:
            metrics = compute_scenario_metrics(seed, atoms)
            proof = [atom.outcome_label for atom in self._atoms if atom.source_id in metrics.active_rules]
            candidates.append(ArbitrageCandidate(seed, metrics, proof))
            for _ in range(fuzz_budget):
                mutated = self._mutate(seed)
                metrics_mut = compute_scenario_metrics(mutated, atoms)
                proof_mut = [
                    atom.outcome_label for atom in self._atoms if atom.source_id in metrics_mut.active_rules
                ]
                candidates.append(ArbitrageCandidate(mutated, metrics_mut, proof_mut))

        top_candidates = sorted(candidates, key=lambda c: c.metrics.score, reverse=True)[:5]
        if top_candidates:
            golden = top_candidates[0]
        else:
            golden = ArbitrageCandidate({}, compute_scenario_metrics({}, atoms), [])

        dossier_metrics = {
            "value": golden.metrics.value_estimate,
            "entropy": golden.metrics.entropy,
            "kappa": golden.metrics.kappa,
            "risk": golden.metrics.risk,
            "contradiction_probability": 1.0 if golden.metrics.contradiction else 0.0,
            "score": golden.metrics.score,
        }
        risk_flags = []
        if golden.metrics.risk > 0.6:
            risk_flags.append("High ambiguity relative to dominant outcome")
        if golden.metrics.contradiction:
            risk_flags.append("Scenario leads to inconsistent obligations")

        return ArbitrageDossier(
            golden_scenario=golden.scenario,
            metrics=dossier_metrics,
            proof_trace=golden.proof_trace,
            risk_flags=risk_flags,
            candidates=top_candidates,
        )
