from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from potatobacon.tariff.atoms_hts import DUTY_RATES


@dataclass(frozen=True)
class RiskAssessment:
    risk_score: int
    defensibility_grade: str
    risk_reasons: List[str]


def changed_fact_keys(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> set[str]:
    keys = set(baseline.keys()) | set(optimized.keys())
    return {key for key in keys if baseline.get(key) != optimized.get(key)}


def is_threshold_fact(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_gt_50") or lowered.endswith("_lt") or lowered.endswith("_ge") or lowered.endswith("_le")


def _get_atom_value(atom: Any, key: str, default: Any = None) -> Any:
    if isinstance(atom, dict):
        return atom.get(key, default)
    return getattr(atom, key, default)


def count_non_tariff_atoms(active_atoms: List[Any]) -> int:
    non_tariff = 0
    for atom in active_atoms:
        rule_type = str(_get_atom_value(atom, "rule_type", "")).upper()
        source_id = _get_atom_value(atom, "source_id", "")
        if rule_type and rule_type != "TARIFF":
            non_tariff += 1
            continue
        if source_id not in DUTY_RATES:
            non_tariff += 1
    return non_tariff


def count_atoms(active_atoms: List[Any]) -> int:
    return len(active_atoms)


def _compute_defensibility_grade(score: int) -> str:
    if score < 30:
        return "A"
    if score < 60:
        return "B"
    return "C"


def assess_tariff_risk(
    baseline_facts: Dict[str, Any],
    optimized_facts: Dict[str, Any],
    baseline_active_atoms: List[Any],
    optimized_active_atoms: List[Any],
    baseline_duty_rate: float,
    optimized_duty_rate: float,
) -> RiskAssessment:
    score = 0
    reasons: List[str] = []

    changed_keys = changed_fact_keys(baseline_facts, optimized_facts)

    # R1: Threshold/Boundary manipulation
    threshold_hits = sum(1 for key in changed_keys if is_threshold_fact(key))
    if threshold_hits:
        score += min(30, 15 + 5 * (threshold_hits - 1))
        reasons.append("Classification hinges on boundary threshold; may trigger audit scrutiny.")

    # R2: Mutation magnitude
    if changed_keys:
        score += min(25, 5 * len(changed_keys))
        reasons.append("Requires multiple product changes; may be harder to implement/defend.")

    # R3: Reliance on bridging rules / notes
    bridging_rules = count_non_tariff_atoms(optimized_active_atoms)
    if bridging_rules:
        score += min(20, 8 + 2 * max(bridging_rules - 1, 0))
        reasons.append("Outcome depends on interpretive note/classification bridge.")

    # R4: Rule chain complexity
    optimized_chain_length = count_atoms(optimized_active_atoms)
    if optimized_chain_length:
        score += min(15, max(0, optimized_chain_length - 3) * 2)
        reasons.append("Longer rule chain increases interpretive surface area; document assumptions.")

    # R5: Novelty of outcome via duty swing
    duty_delta = baseline_duty_rate - optimized_duty_rate
    if duty_delta > 20:
        score += 15
        reasons.append("Large duty swing may attract compliance review; ensure documentation.")
    elif duty_delta > 10:
        score += 10
        reasons.append("Material duty reduction should be documented for audit defense.")

    if baseline_facts.get("ad_cvd_possible") or optimized_facts.get("ad_cvd_possible"):
        score += 10
        reasons.append("Potential AD/CVD exposure; requires specialist review.")

    score = max(0, min(100, score))
    grade = _compute_defensibility_grade(score)

    # Trim to at most 5 reasons while preserving determinism
    reasons = reasons[:5] if reasons else ["Low-risk change with clear provenance."]

    return RiskAssessment(risk_score=score, defensibility_grade=grade, risk_reasons=reasons)
