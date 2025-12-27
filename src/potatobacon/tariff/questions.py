from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.fact_requirements import FactRequirementRegistry
from potatobacon.tariff.models import BaselineCandidateModel
from potatobacon.tariff.sku_models import (
    ConditionalPathwayModel,
    IntakeBundleItemModel,
    IntakeBundleModel,
    MissingFactsPackageModel,
    QuestionItemModel,
)
from potatobacon.tariff.models import TariffSuggestionItemModel


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


class BundleAggregator:
    """Aggregate missing facts into evidence-first intake bundles."""

    _LABELS = {
        "bom_csv": "BOM CSV",
        "bom_csv_origin_column": "BOM CSV",
        "bom_csv_section_highlight": "BOM CSV",
        "spec_sheet": "Technical Spec",
        "spec_sheet_pdf": "Technical Spec",
        "connector_spec_sheet": "Technical Spec",
        "safety_datasheet_pdf": "Technical Spec",
        "material_certificate_pdf": "Material Composition Proof",
        "lab_certificate_pdf": "Lab Certificate",
        "composition_test": "Lab Certificate",
        "customs_ruling_pdf": "Customs Ruling",
        "manufacturing_step_logs": "Manufacturing Step Logs",
        "labor_cost_summary": "Labor Cost Summary",
        "sub_supplier_cert_origin": "Sub-supplier Origin Certificates",
    }

    def __init__(self, requirement_registry: FactRequirementRegistry | None = None) -> None:
        self._registry = requirement_registry or FactRequirementRegistry()

    def _label_for(self, evidence_type: str) -> str:
        if evidence_type in self._LABELS:
            return self._LABELS[evidence_type]
        return evidence_type.replace("_", " ").title()

    def build(
        self,
        *,
        conditional_pathways: Sequence[ConditionalPathwayModel],
        suggestions: Sequence[TariffSuggestionItemModel],
        fact_savings: Mapping[str, float | None],
        origin_fact_gaps: Sequence[str] = (),
    ) -> IntakeBundleModel:
        evidence_to_facts: Dict[str, set[str]] = {}

        for pathway in conditional_pathways:
            for fact_key in pathway.missing_facts:
                requirement = self._registry.describe(fact_key)
                for evidence_type in requirement.evidence_types:
                    evidence_to_facts.setdefault(evidence_type, set()).add(fact_key)

        for suggestion in suggestions:
            for fact_key in suggestion.fact_gaps:
                requirement = self._registry.describe(fact_key)
                for evidence_type in requirement.evidence_types:
                    evidence_to_facts.setdefault(evidence_type, set()).add(fact_key)

        for fact_key in origin_fact_gaps:
            requirement = self._registry.describe(fact_key)
            for evidence_type in requirement.evidence_types:
                evidence_to_facts.setdefault(evidence_type, set()).add(fact_key)

        items: List[IntakeBundleItemModel] = []
        for evidence_type in sorted(evidence_to_facts.keys()):
            fact_keys = sorted(evidence_to_facts[evidence_type])
            potential = [
                value
                for fact_key in fact_keys
                for value in [fact_savings.get(fact_key)]
                if value is not None
            ]
            items.append(
                IntakeBundleItemModel(
                    request_label=self._label_for(evidence_type),
                    evidence_types=[evidence_type],
                    fact_keys=fact_keys,
                    potential_savings_unlocked=max(potential) if potential else None,
                )
            )

        return IntakeBundleModel(items=items)
