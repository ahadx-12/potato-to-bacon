"""Scenario-level metrics for CALE-LAW using Z3 semantics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from potatobacon.law.ambiguity_entropy import normalized_entropy
from potatobacon.law.solver_z3 import PolicyAtom, check_scenario


@dataclass(slots=True)
class ScenarioMetrics:
    probabilities: Dict[str, float]
    entropy: float
    contradiction: bool
    kappa: float
    value_estimate: float
    risk: float
    score: float
    active_rules: List[str]


def _applicable_outcomes(
    scenario: Mapping[str, bool], atoms: Sequence[PolicyAtom]
) -> List[PolicyAtom]:
    sat, active = check_scenario(scenario, atoms)
    if not sat:
        return active
    return active


def _dirichlet_counts(labels: Iterable[str], alpha: float = 0.1) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for label in labels:
        counts[label] = counts.get(label, alpha) + 1.0
    for key in list(counts):
        if counts[key] == 0:
            counts[key] = alpha
    return counts


def _outcome_probabilities(active_atoms: Sequence[PolicyAtom]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    modality_weight = {"OBLIGE": 1.2, "FORBID": 1.0, "PERMIT": 0.8}
    for atom in active_atoms:
        label = atom.outcome_label
        modality = atom.outcome.get("modality", "PERMIT").upper()
        weight = modality_weight.get(modality, 1.0)
        weights[label] = weights.get(label, 0.0) + weight
    if not weights:
        return {}
    counts = _dirichlet_counts(weights.keys())
    for label, weight in weights.items():
        counts[label] = counts.get(label, 0.0) + weight
    total = sum(counts.values())
    return {label: value / total for label, value in counts.items()}


def compute_scenario_metrics(
    scenario: Mapping[str, bool], atoms: Sequence[PolicyAtom], alpha: float = 1.0, beta: float = 1.0
) -> ScenarioMetrics:
    """Compute entropy, contradiction, and arbitrage heuristics for a scenario."""

    is_sat, active_atoms = check_scenario(scenario, atoms)
    probabilities = _outcome_probabilities(active_atoms)
    entropy = normalized_entropy(probabilities.values()) if probabilities else 0.0
    dominant = max(probabilities.values(), default=0.0)
    risk = max(0.0, 1.0 - dominant)

    # Value heuristic: spread between best and median likelihoods
    sorted_probs = sorted(probabilities.values(), reverse=True)
    if len(sorted_probs) >= 2:
        value_estimate = max(0.0, sorted_probs[0] - sorted_probs[1])
    elif sorted_probs:
        value_estimate = sorted_probs[0]
    else:
        value_estimate = 0.0

    score = (value_estimate ** alpha) * (entropy ** beta) * (1.0 - risk)

    # Cohen's kappa proxy: agreement vs uniform baseline
    expected = 1.0 / max(len(probabilities), 1) if probabilities else 0.0
    kappa = 0.0
    if probabilities:
        kappa = (dominant - expected) / (1.0 - expected) if expected < 1.0 else 0.0
        kappa = max(0.0, min(1.0, kappa))

    return ScenarioMetrics(
        probabilities=probabilities,
        entropy=float(entropy),
        contradiction=not is_sat,
        kappa=float(kappa),
        value_estimate=float(value_estimate),
        risk=float(risk),
        score=float(score),
        active_rules=[atom.source_id for atom in active_atoms],
    )


def _random_scenario(predicates: Sequence[str]) -> Dict[str, bool]:
    return {pred: bool(random.getrandbits(1)) for pred in predicates}


def sample_scenarios(atoms: Sequence[PolicyAtom], sample_size: int = 20) -> List[Dict[str, bool]]:
    """Sample fact assignments implied by the manifest atoms."""

    seen_predicates: MutableMapping[str, None] = {}
    for atom in atoms:
        for literal in atom.guard:
            name = literal[1:] if literal.startswith("¬") else literal
            seen_predicates.setdefault(name, None)
    predicates = list(seen_predicates.keys()) or ["default_fact"]
    return [_random_scenario(predicates) for _ in range(sample_size)]


def batch_metrics(atoms: Sequence[PolicyAtom], sample_size: int = 20) -> Dict[str, float]:
    """Aggregate contradiction probability and mean entropy over samples."""

    scenarios = sample_scenarios(atoms, sample_size=sample_size)
    if not scenarios:
        return {"contradiction_probability": 0.0, "mean_entropy": 0.0}

    contradiction_count = 0
    entropy_accum = 0.0
    for scenario in scenarios:
        metrics = compute_scenario_metrics(scenario, atoms)
        if metrics.contradiction:
            contradiction_count += 1
        entropy_accum += metrics.entropy

    return {
        "contradiction_probability": float(contradiction_count) / float(len(scenarios)),
        "mean_entropy": float(entropy_accum) / float(len(scenarios)),
    }
