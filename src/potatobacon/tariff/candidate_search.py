"""Deterministic classification candidate search for tariff scenarios."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from potatobacon.law.solver_z3 import PolicyAtom, analyze_scenario
from potatobacon.tariff.models import BaselineCandidateModel


def _fact_key(literal: str) -> str:
    return literal[1:] if literal.startswith("¬") else literal


def _infer_atom_categories(guard: Sequence[str]) -> set[str]:
    categories: set[str] = set()
    for literal in guard:
        fact_key = _fact_key(literal)
        if fact_key in {"product_type_electronics"} or fact_key.startswith("electronics_") or fact_key in {
            "contains_pcb",
            "contains_battery",
        }:
            categories.add("electronics")
        if fact_key in {"product_type_chassis_bolt", "is_fastener"} or fact_key.startswith("fastener_"):
            categories.add("fastener")
        if fact_key.startswith("surface_contact_") or fact_key.startswith("upper_material_") or fact_key.startswith(
            "outer_sole_material_"
        ):
            categories.add("footwear")
        if fact_key == "product_type_apparel_textile" or fact_key.startswith("textile_") or fact_key.startswith("fiber_"):
            categories.add("apparel_textile")
    return categories


def _provenance_for_atom(atom: PolicyAtom, scenario_label: str) -> Dict[str, str]:
    return {
        "scenario": scenario_label,
        "source_id": atom.source_id,
        "statute": getattr(atom, "statute", ""),
        "section": getattr(atom, "section", ""),
        "text": getattr(atom, "text", ""),
        "jurisdiction": atom.outcome.get("jurisdiction", ""),
    }


def _evaluate_guard(facts: Mapping[str, object], guard: Sequence[str]) -> tuple[list[str], int, bool]:
    missing: list[str] = []
    satisfied = 0
    contradiction = False

    for literal in guard:
        negated = literal.startswith("¬")
        fact_key = literal[1:] if negated else literal
        if fact_key not in facts or facts.get(fact_key) is None:
            missing.append(fact_key)
            continue
        fact_value = bool(facts.get(fact_key))
        if (fact_value and negated) or (not fact_value and not negated):
            contradiction = True
            break
        satisfied += 1

    return missing, satisfied, contradiction


def generate_baseline_candidates(
    facts: Mapping[str, object],
    atoms: Iterable[PolicyAtom],
    duty_rates: Mapping[str, float],
    max_candidates: int = 5,
) -> List[BaselineCandidateModel]:
    """Enumerate and rank duty-bearing atoms applicable to *facts*.

    Candidates are ranked deterministically by lower duty rate, more specific
    guard length, and then lexicographically by ``source_id``. Missing facts
    diminish confidence but do not discard otherwise compatible candidates.
    """

    sat, active_atoms, unsat_core = analyze_scenario(facts, list(atoms))
    active_duty_atoms = [atom for atom in active_atoms if atom.source_id in duty_rates]
    active_lookup = {atom.source_id for atom in active_duty_atoms}

    ranked: list[tuple[float, int, str, BaselineCandidateModel]] = []
    declared_category = str(facts.get("product_category") or "").lower()

    for atom in atoms:
        if atom.source_id not in duty_rates:
            continue

        atom_categories = _infer_atom_categories(atom.guard)
        if declared_category and atom_categories and declared_category not in atom_categories:
            continue

        missing, satisfied, contradiction = _evaluate_guard(facts, atom.guard)
        if contradiction:
            continue
        if atom.guard and satisfied == 0 and len(missing) == len(atom.guard):
            # No overlap between guard and provided facts; skip unrelated candidates
            continue

        guard_length = len(atom.guard)
        duty_rate = float(duty_rates[atom.source_id])
        base_confidence = 0.6 if guard_length == 0 else satisfied / guard_length
        if missing:
            penalty = (len(missing) / max(1, guard_length)) * 0.3
            base_confidence = max(0.05, base_confidence * (1 - penalty))
        if not sat:
            base_confidence = max(0.05, base_confidence * 0.5)
        base_confidence = min(1.0, base_confidence)

        provenance_label = "baseline" if atom.source_id in active_lookup and sat and not missing else "candidate"
        provenance_chain = [_provenance_for_atom(atom, provenance_label)]

        candidate = BaselineCandidateModel(
            candidate_id=atom.source_id,
            active_codes=sorted({a.source_id for a in active_duty_atoms}) if active_lookup else [atom.source_id],
            duty_rate=duty_rate,
            provenance_chain=provenance_chain,
            confidence=base_confidence,
            missing_facts=sorted(set(missing)),
            compliance_flags={
                "requires_ruling_review": bool(unsat_core),
                "guard_satisfied": atom.source_id in active_lookup and not missing,
            },
        )

        ranked.append((duty_rate, -guard_length, atom.source_id, candidate))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [candidate for _, _, _, candidate in ranked[:max_candidates]]
