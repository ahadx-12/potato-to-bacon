from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.fact_requirements import FactRequirementRegistry
from potatobacon.tariff.models import BaselineCandidateModel
from potatobacon.tariff.sku_models import MissingFactsPackageModel, QuestionItemModel


def _fact_from_literal(literal: str) -> str:
    return literal[1:] if literal.startswith("¬") else literal


def _atom_descriptor(atom: PolicyAtom) -> str:
    section = getattr(atom, "section", "")
    return f"{atom.source_id}:{section}" if section else atom.source_id


def _why_needed(fact_key: str, atoms: Sequence[PolicyAtom], levers: Sequence[str]) -> str:
    rationale: List[str] = []
    if atoms:
        impacted = ", ".join(_atom_descriptor(atom) for atom in atoms)
        rationale.append(f"Required for candidate atom(s): {impacted}.")
    if levers:
        rationale.append(f"Enables optimization lever(s): {', '.join(sorted(levers))}.")
    if not rationale and fact_key.startswith("origin"):
        rationale.append("Origin drives trade remedies, FTA eligibility, and valuation.")
    if not rationale:
        rationale.append("Needed to resolve duty-bearing classification confidently.")
    return " ".join(rationale)


def generate_missing_fact_questions(
    *,
    law_context: str,
    atoms: Iterable[PolicyAtom],
    compiled_facts: Mapping[str, object],
    candidates: Sequence[BaselineCandidateModel],
    lever_requirements: Mapping[str, Sequence[str]] | None = None,
) -> MissingFactsPackageModel:
    """Generate deterministic questions for missing facts from candidate search."""

    best_confidence = max((candidate.confidence for candidate in candidates), default=0.0)
    confidence_floor = max(0.05, min(best_confidence - 0.35, 0.25))
    shortlisted = [candidate for candidate in candidates if candidate.confidence >= confidence_floor]

    requirement_registry = FactRequirementRegistry()
    lever_requirements = lever_requirements or {}
    lever_missing_keys = set(lever_requirements.keys())
    candidate_ids = {candidate.candidate_id for candidate in shortlisted}
    candidate_missing_keys = {fact for candidate in shortlisted for fact in candidate.missing_facts}
    atoms_list = list(atoms)
    atoms_by_id = {atom.source_id: atom for atom in atoms_list if atom.source_id in candidate_ids}
    origin_missing = True
    for key, value in compiled_facts.items():
        if key.startswith("origin_country_") and value:
            origin_missing = False
            break
    if origin_missing and compiled_facts.get("requires_origin_data"):
        candidate_missing_keys.add("origin_country")

    fact_to_atoms: dict[str, List[PolicyAtom]] = {}
    for atom in atoms_by_id.values():
        for literal in atom.guard:
            fact_key = _fact_from_literal(literal)
            if fact_key in candidate_missing_keys:
                fact_to_atoms.setdefault(fact_key, []).append(atom)

    missing_keys = sorted(set(candidate_missing_keys).union(lever_missing_keys))
    questions: List[QuestionItemModel] = []
    for fact_key in missing_keys:
        related_atoms = sorted(
            fact_to_atoms.get(fact_key, []),
            key=lambda atom: (_atom_descriptor(atom), getattr(atom, "text", "")),
        )
        lever_ids = sorted({lever_id for lever_id in lever_requirements.get(fact_key, [])})
        requirement = requirement_registry.describe(fact_key)
        questions.append(
            QuestionItemModel(
                fact_key=fact_key,
                question=requirement.render_question(),
                why_needed=_why_needed(fact_key, related_atoms, lever_ids),
                accepted_evidence_types=list(requirement.evidence_types),
                measurement_hint=requirement.measurement_hint,
                candidate_rules_affected=[_atom_descriptor(atom) for atom in related_atoms],
                lever_ids_affected=lever_ids,
                blocks_classification=fact_key in candidate_missing_keys,
                blocks_optimization=bool(lever_ids or fact_key in candidate_missing_keys),
            )
        )

    return MissingFactsPackageModel(missing_facts=missing_keys, questions=questions)
