from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.models import BaselineCandidateModel
from potatobacon.tariff.sku_models import MissingFactsPackageModel, QuestionItemModel


def _fact_from_literal(literal: str) -> str:
    return literal[1:] if literal.startswith("¬") else literal


def _atom_descriptor(atom: PolicyAtom) -> str:
    section = getattr(atom, "section", "")
    return f"{atom.source_id}:{section}" if section else atom.source_id


def _question_text(fact_key: str) -> str:
    if fact_key.startswith("origin_country") or fact_key.startswith("origin_"):
        return "What is the country of origin for the SKU and its key components?"
    if "voltage_rating" in fact_key:
        return "What voltage/current rating applies to the cable or connector assembly?"
    if "insulated_conductors" in fact_key:
        return "Are the conductors insulated/jacketed as part of the assembly?"
    if "has_connectors" in fact_key:
        return "Does the assembly terminate with defined connectors?"
    if "is_cable_assembly" in fact_key:
        return "Is this SKU sold as a complete cable or harness assembly?"
    if "surface_contact_textile_gt_50" in fact_key:
        return "Does textile material cover more than 50% of the outsole contact surface?"
    if "surface_contact_rubber_gt_50" in fact_key:
        return "Is rubber or plastic covering more than 50% of the outsole contact surface?"
    if "material" in fact_key:
        return f"Confirm the material evidence for '{fact_key}'."
    if "electronics" in fact_key:
        return f"Provide design evidence to resolve electronics attribute '{fact_key}'."
    return f"Provide a definitive value for '{fact_key}'."


def _evidence_hints(fact_key: str) -> List[str]:
    if fact_key.startswith("origin_country") or fact_key.startswith("origin_"):
        return ["Certificate of origin", "Commercial invoice", "Bill of materials origin column"]
    if "voltage_rating" in fact_key:
        return ["Electrical rating sheet", "Connector spec (USB/HDMI class)", "Safety datasheet with voltage/current"]
    if "insulated_conductors" in fact_key:
        return ["Cable cross-section photo", "Jacket material declaration", "Harness drawing noting insulation"]
    if "has_connectors" in fact_key:
        return ["Connector drawings", "Harness pinout", "BOM line items showing connector part numbers"]
    if "is_cable_assembly" in fact_key:
        return ["Harness or cable assembly drawing", "BOM section highlighting assembly form"]
    if "material" in fact_key or "surface_contact" in fact_key:
        return ["BOM line items", "Material certificates", "Photos or measurements"]
    if "electronics" in fact_key:
        return ["Assembly drawings", "Harness or connector photos", "BOM electronics markers"]
    return ["BOM line items", "Technical spec sheet"]


def _why_it_matters(fact_key: str, atoms: Sequence[PolicyAtom]) -> str:
    if atoms:
        impacted = ", ".join(_atom_descriptor(atom) for atom in atoms)
        return f"Required to evaluate tariff rules: {impacted}."
    if fact_key.startswith("origin"):
        return "Origin drives trade remedies and FTA eligibility."
    return "Needed to resolve duty-bearing classification confidently."


def generate_missing_fact_questions(
    *,
    law_context: str,
    atoms: Iterable[PolicyAtom],
    compiled_facts: Mapping[str, object],
    candidates: Sequence[BaselineCandidateModel],
) -> MissingFactsPackageModel:
    """Generate deterministic questions for missing facts from candidate search."""

    best_confidence = max((candidate.confidence for candidate in candidates), default=0.0)
    confidence_floor = max(0.05, min(best_confidence - 0.35, 0.25))
    shortlisted = [candidate for candidate in candidates if candidate.confidence >= confidence_floor]

    candidate_ids = {candidate.candidate_id for candidate in shortlisted}
    missing_keys = {fact for candidate in shortlisted for fact in candidate.missing_facts}
    atoms_list = list(atoms)
    atoms_by_id = {atom.source_id: atom for atom in atoms_list if atom.source_id in candidate_ids}
    origin_missing = True
    for key, value in compiled_facts.items():
        if key.startswith("origin_country_") and value:
            origin_missing = False
            break
    if origin_missing and compiled_facts.get("requires_origin_data"):
        missing_keys.add("origin_country")

    fact_to_atoms: dict[str, List[PolicyAtom]] = {}
    for atom in atoms_by_id.values():
        for literal in atom.guard:
            fact_key = _fact_from_literal(literal)
            if fact_key in missing_keys:
                fact_to_atoms.setdefault(fact_key, []).append(atom)

    missing_sorted = sorted(missing_keys)
    questions: List[QuestionItemModel] = []
    for fact_key in missing_sorted:
        related_atoms = sorted(
            fact_to_atoms.get(fact_key, []),
            key=lambda atom: (_atom_descriptor(atom), getattr(atom, "text", "")),
        )
        questions.append(
            QuestionItemModel(
                fact_key=fact_key,
                question=_question_text(fact_key),
                why_it_matters=_why_it_matters(fact_key, related_atoms),
                evidence_requested=_evidence_hints(fact_key),
                candidate_rules_affected=[_atom_descriptor(atom) for atom in related_atoms],
            )
        )

    return MissingFactsPackageModel(missing_facts=missing_sorted, questions=questions)
