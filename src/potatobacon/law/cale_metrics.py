"""Scenario-level metrics for CALE-LAW using Z3 semantics.

Each metric is designed to be empirically inspectable:
- ``probabilities`` and ``entropy`` follow a Dirichlet-smoothed distribution of
  active rule outcomes.
- ``contradiction`` and ``contradiction_probability`` are derived directly from
  Z3 SAT/UNSAT checks across nearby scenarios instead of static heuristics.
- ``value_components`` model pre-/post-tax cash flows with explicit loss
  handling and transparent tax-rate blending.
- ``risk_components`` clamp constituent risks to [0, 1] for calibration
  stability and align the aggregate ``risk`` term with observed entropy and
  enforcement tension.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from potatobacon.law.ambiguity_entropy import normalized_entropy
from potatobacon.law.solver_z3 import PolicyAtom, check_scenario


@dataclass(slots=True)
class ScenarioMetrics:
    """Computed metrics for a single scenario.

    - ``probabilities``: Dirichlet-smoothed probability mass over active outcomes.
    - ``entropy``: Normalized entropy of the outcome distribution.
    - ``contradiction``/``contradiction_probability``: Boolean SAT flag and
      empirical ratio of UNSAT checks across local perturbations.
    - ``kappa``: Agreement vs uniform expectation for the dominant outcome.
    - ``value_estimate``: Spread between dominant and runner-up outcome weights.
    - ``risk``/``risk_components``: Clamped constituent risks (enforcement,
      ambiguity, treaty mismatch) and their aggregate.
    - ``score``/``score_components``: Composite value × entropy × (1 - risk)
      with tunable ``alpha``/``beta`` and seed traceability.
    - ``value_components``: Gross income, derived tax rates, tax liability, and
      net-after-tax (non-negative unless explicit loss facts are present).
    """

    probabilities: Dict[str, float]
    entropy: float
    contradiction: bool
    contradiction_probability: float
    kappa: float
    value_estimate: float
    risk: float
    score: float
    active_rules: List[str]
    value_components: Dict[str, float]
    risk_components: Dict[str, float]
    score_components: Dict[str, float]
    alpha: float
    beta: float
    seed: int | None


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


def _local_contradiction_probability(
    scenario: Mapping[str, bool], atoms: Sequence[PolicyAtom], base_sat: bool
) -> float:
    """Estimate contradiction likelihood by probing nearby scenarios.

    The estimator compares the SAT/UNSAT ratio for the provided ``scenario`` and
    a handful of single-bit flips. This anchors the probability to concrete Z3
    outcomes rather than a static heuristic.
    """

    if not atoms:
        return 0.0

    trials = 1
    contradiction_count = 0 if base_sat else 1
    for key, value in list(scenario.items())[:5]:
        mutated = dict(scenario)
        mutated[key] = not value
        is_sat, _ = check_scenario(mutated, atoms)
        if not is_sat:
            contradiction_count += 1
        trials += 1
    return max(0.0, min(1.0, contradiction_count / max(trials, 1)))


def compute_scenario_metrics(
    scenario: Mapping[str, bool],
    atoms: Sequence[PolicyAtom],
    alpha: float = 1.0,
    beta: float = 1.0,
    seed: int | None = None,
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

    # Value transparency: derive components based on active rules and entropy
    active_jurisdictions: Dict[str, int] = {}
    for atom in active_atoms:
        jurisdiction = atom.outcome.get("jurisdiction", "Unknown") or "Unknown"
        active_jurisdictions[jurisdiction] = active_jurisdictions.get(jurisdiction, 0) + 1

    gross_income = float(max(len(scenario), 1) * 100000.0)
    effective_tax_rate_base = max(0.0, min(1.0, 0.25 + (1.0 - dominant) * 0.5))
    value_components: Dict[str, float] = {
        "gross_income": gross_income,
    }
    for jurisdiction, count in active_jurisdictions.items():
        weight = count / max(len(active_atoms), 1)
        key = f"effective_tax_rate_{jurisdiction.lower().replace(' ', '_').replace('.', '_')}"
        value_components[key] = max(0.0, min(1.0, effective_tax_rate_base * weight + entropy * 0.1))
    blended_tax_rate = max(value_components.values()) if len(value_components) > 1 else effective_tax_rate_base
    tax_liability = gross_income * blended_tax_rate
    net_after_tax = gross_income - tax_liability
    if net_after_tax < 0 and not scenario.get("loss", False) and not scenario.get("net_loss", False):
        net_after_tax = 0.0
    value_components["tax_liability"] = tax_liability
    value_components["net_after_tax"] = net_after_tax

    # Risk transparency: trace drivers behind the scalar risk
    enforcement_risk = max(0.0, min(1.0, risk + (0.2 if not is_sat else 0.0)))
    ambiguity_risk = max(0.0, min(1.0, entropy))
    treaty_mismatch_risk = max(0.0, min(1.0, len(active_jurisdictions) * 0.1 + (1.0 - dominant) * 0.3))
    risk_components = {
        "enforcement_risk": enforcement_risk,
        "ambiguity_risk": ambiguity_risk,
        "treaty_mismatch_risk": treaty_mismatch_risk,
    }
    risk = max(0.0, min(1.0, max(risk, sum(risk_components.values()) / max(len(risk_components), 1))))

    value_term = value_estimate**alpha
    entropy_term = entropy**beta
    risk_term = 1.0 - risk
    score = value_term * entropy_term * risk_term
    score_components = {
        "value_term": value_term,
        "entropy_term": entropy_term,
        "risk_term": risk_term,
        "alpha": alpha,
        "beta": beta,
        "seed": int(seed) if seed is not None else -1,
    }

    # Cohen's kappa proxy: agreement vs uniform baseline
    expected = 1.0 / max(len(probabilities), 1) if probabilities else 0.0
    kappa = 0.0
    if probabilities:
        kappa = (dominant - expected) / (1.0 - expected) if expected < 1.0 else 0.0
        kappa = max(0.0, min(1.0, kappa))

    contradiction_probability = _local_contradiction_probability(scenario, atoms, base_sat=is_sat)

    return ScenarioMetrics(
        probabilities=probabilities,
        entropy=float(entropy),
        contradiction=not is_sat,
        contradiction_probability=float(contradiction_probability),
        kappa=float(kappa),
        value_estimate=float(value_estimate),
        risk=float(risk),
        score=float(score),
        active_rules=[atom.source_id for atom in active_atoms],
        value_components=value_components,
        risk_components=risk_components,
        score_components=score_components,
        alpha=float(alpha),
        beta=float(beta),
        seed=seed,
    )


def _random_scenario(predicates: Sequence[str], rng: random.Random | None = None) -> Dict[str, bool]:
    rng = rng or random
    return {pred: bool(rng.getrandbits(1)) for pred in predicates}


def sample_scenarios(
    atoms: Sequence[PolicyAtom], sample_size: int = 20, rng: random.Random | None = None
) -> List[Dict[str, bool]]:
    """Sample fact assignments implied by the manifest atoms."""

    seen_predicates: MutableMapping[str, None] = {}
    for atom in atoms:
        for literal in atom.guard:
            name = literal[1:] if literal.startswith("¬") else literal
            seen_predicates.setdefault(name, None)
    predicates = list(seen_predicates.keys()) or ["default_fact"]
    return [_random_scenario(predicates, rng=rng) for _ in range(sample_size)]


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
