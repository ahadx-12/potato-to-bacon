from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.tariff.atom_utils import atom_provenance
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
from potatobacon.tariff.candidate_search import generate_baseline_candidates
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.fact_requirements import FactRequirementRegistry
from potatobacon.tariff.levers import applicable_levers, lever_library
from potatobacon.tariff.models import BaselineCandidateModel, FactEvidenceModel, TariffSuggestionItemModel
from potatobacon.tariff.mutation_generator import baseline_facts_from_profile, infer_product_profile
from potatobacon.tariff.normalizer import normalize_compiled_facts, validate_minimum_inputs
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.product_schema import ProductCategory
from potatobacon.tariff.questions import generate_missing_fact_questions
from potatobacon.tariff.risk import assess_tariff_risk
from potatobacon.tariff.sku_models import (
    BaselineAssignmentModel,
    ConditionalPathwayModel,
    FactOverrideModel,
    SKUDossierBaselineModel,
    SKUDossierOptimizedModel,
    TariffSkuDossierV2Model,
    build_sku_metadata_snapshot,
)
from potatobacon.tariff.sku_store import SKUStore, get_default_sku_store
from potatobacon.tariff.suggest import (
    _compute_savings,
    _defensibility_grade,
    _duty_atoms,
    _evaluate_scenario,
    _record_proof,
    _sort_key,
)


def _merge_facts(baseline: Dict[str, Any], compiled: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(baseline)
    for key, value in compiled.items():
        if value or key not in merged:
            merged[key] = value
    return merged


def _apply_fact_overrides(
    merged_facts: Dict[str, Any],
    overrides: Dict[str, FactOverrideModel],
) -> tuple[Dict[str, Any], Dict[str, Any], list[FactEvidenceModel]]:
    updated = dict(merged_facts)
    overrides_payload: Dict[str, Any] = {}
    override_evidence: list[FactEvidenceModel] = []
    for key in sorted(overrides.keys()):
        override = overrides[key]
        updated[key] = override.value
        overrides_payload[key] = override.serializable_dict()
        override_evidence.append(
            FactEvidenceModel(
                fact_key=key,
                value=override.value,
                confidence=override.confidence if override.confidence is not None else 0.95,
                evidence=[],
                derived_from=[override.source],
                risk_reason="Manual override applied via analysis session",
            )
        )
    override_evidence.sort(
        key=lambda item: (
            item.fact_key,
            str(item.value),
            item.confidence,
            len(item.evidence),
        )
    )
    return updated, overrides_payload, override_evidence


def _evidence_pack(
    *,
    product_spec: Any,
    compiled_facts: Dict[str, Any],
    fact_evidence: Any,
    extraction_evidence: Any,
    sku_metadata: Dict[str, Any],
    analysis_session_id: str | None = None,
    attached_evidence_ids: list[str] | None = None,
    fact_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    pack: Dict[str, Any] = {
        "product_spec": product_spec,
        "compiled_facts": compiled_facts,
        "fact_evidence": fact_evidence,
        "extraction_evidence": extraction_evidence,
    }
    if fact_overrides:
        pack["fact_overrides"] = fact_overrides
    if analysis_session_id or attached_evidence_ids:
        session_pack: Dict[str, Any] = {}
        if analysis_session_id:
            session_pack["session_id"] = analysis_session_id
        if attached_evidence_ids:
            session_pack["attached_evidence_ids"] = sorted(attached_evidence_ids)
        pack["analysis_session"] = session_pack
    if sku_metadata:
        pack["sku_metadata"] = sku_metadata
    return pack


def _lever_requirement_gaps(spec: Any, facts: Mapping[str, object]) -> dict[str, list[str]]:
    """Return missing fact requirements keyed by fact needed to activate a lever."""

    category = getattr(spec, "product_category", None)
    category_value = category.value if category else "other"
    fact_to_levers: dict[str, set[str]] = {}
    for lever in lever_library():
        if "all" not in lever.category_scope and category_value not in lever.category_scope:
            continue
        conflicting = False
        missing_keys: list[str] = []
        for key, expected in lever.required_facts.items():
            if key in facts and expected is not None and bool(facts[key]) != bool(expected):
                conflicting = True
                break
            if facts.get(key) is None:
                missing_keys.append(key)
        if conflicting or not missing_keys:
            continue
        for key in missing_keys:
            fact_to_levers.setdefault(key, set()).add(lever.lever_id)
    return {key: sorted(values) for key, values in fact_to_levers.items()}


def _baseline_candidates(
    normalized_facts: Dict[str, Any], atoms: Any, duty_rates: Dict[str, float]
) -> tuple[list[BaselineCandidateModel], Any]:
    baseline_candidates = generate_baseline_candidates(normalized_facts, atoms, duty_rates, max_candidates=5)
    baseline_eval = _evaluate_scenario(atoms, normalized_facts, duty_rates)
    return baseline_candidates, baseline_eval


def _provenance_for_atoms(atoms: Any, scenario_label: str) -> list[Dict[str, Any]]:
    chain: list[Dict[str, Any]] = []
    for atom in atoms or []:
        chain.append(atom_provenance(atom, scenario_label))
    return chain


def _combined_provenance(baseline_atoms: Any, optimized_atoms: Any) -> list[Dict[str, Any]]:
    chain = _provenance_for_atoms(baseline_atoms, "baseline") + _provenance_for_atoms(optimized_atoms, "optimized")
    chain.sort(
        key=lambda item: (
            item.get("source_id", ""),
            item.get("section", ""),
            item.get("text", ""),
            item.get("scenario", ""),
        )
    )
    return chain


def _select_baseline_assignment(
    baseline_eval: Any, baseline_candidates: list[BaselineCandidateModel], duty_rates: Dict[str, float]
) -> BaselineAssignmentModel:
    if getattr(baseline_eval, "duty_atoms", None):
        ranked_atoms = sorted(
            baseline_eval.duty_atoms,
            key=lambda atom: (float(duty_rates.get(atom.source_id, 999.0)), atom.source_id),
        )
        top_atom = ranked_atoms[0]
        candidate_lookup = {cand.candidate_id: cand for cand in baseline_candidates}
        candidate = candidate_lookup.get(top_atom.source_id)
        return BaselineAssignmentModel(
            atom_id=top_atom.source_id,
            duty_rate=float(duty_rates.get(top_atom.source_id, baseline_eval.duty_rate)),
            duty_status=getattr(baseline_eval, "duty_status", None),
            confidence=candidate.confidence if candidate else None,
        )

    eligible = [
        cand
        for cand in baseline_candidates
        if not cand.missing_facts and cand.compliance_flags.get("guard_satisfied", False)
    ]
    eligible.sort(
        key=lambda cand: (
            cand.duty_rate,
            -len(cand.active_codes),
            -(cand.confidence or 0.0),
            cand.candidate_id,
        )
    )
    if eligible:
        best = eligible[0]
        return BaselineAssignmentModel(
            atom_id=best.candidate_id,
            duty_rate=best.duty_rate,
            duty_status=getattr(baseline_eval, "duty_status", None),
            confidence=best.confidence,
        )

    return BaselineAssignmentModel(
        atom_id=None,
        duty_rate=getattr(baseline_eval, "duty_rate", None),
        duty_status=getattr(baseline_eval, "duty_status", None),
        confidence=None,
    )


def _conditional_pathways(
    baseline_candidates: list[BaselineCandidateModel],
    baseline_assignment: BaselineAssignmentModel,
    *,
    requirement_registry: FactRequirementRegistry,
) -> list[ConditionalPathwayModel]:
    base_rate = baseline_assignment.duty_rate
    pathways: list[ConditionalPathwayModel] = []
    for candidate in baseline_candidates:
        if not candidate.missing_facts:
            continue
        if base_rate is not None and candidate.duty_rate >= base_rate:
            continue
        why: list[str] = []
        evidence_types: set[str] = set()
        for fact_key in candidate.missing_facts:
            requirement = requirement_registry.describe(fact_key)
            why.append(requirement.render_question())
            evidence_types.update(requirement.evidence_types)
        pathways.append(
            ConditionalPathwayModel(
                atom_id=candidate.candidate_id,
                duty_rate=candidate.duty_rate,
                missing_facts=list(candidate.missing_facts),
                why_needed=sorted(set(why)),
                accepted_evidence_types=sorted(evidence_types),
            )
        )
    pathways.sort(key=lambda path: (path.duty_rate, path.atom_id))
    return pathways


def build_sku_dossier_v2(
    sku_id: str,
    *,
    law_context: Optional[str] = None,
    evidence_requested: bool = False,
    optimize: bool = True,
    risk_penalize: Optional[bool] = None,  # reserved for future risk-aware ranking
    store: SKUStore | None = None,
    fact_overrides: Dict[str, FactOverrideModel] | None = None,
    attached_evidence_ids: list[str] | None = None,
    session_id: str | None = None,
) -> TariffSkuDossierV2Model:
    """Generate a SKU-first dossier covering baseline, optimization, and questions."""

    sku_store = store or get_default_sku_store()
    record = sku_store.get(sku_id)
    if not record:
        raise KeyError(sku_id)

    overrides_input = fact_overrides or {}
    override_models: Dict[str, FactOverrideModel] = {}
    for key, value in overrides_input.items():
        override_models[key] = value if isinstance(value, FactOverrideModel) else FactOverrideModel(**value)
    attached_evidence_ids = sorted({eid for eid in attached_evidence_ids or []})

    bom_structured = record.bom_json
    if bom_structured is None and record.bom_csv:
        bom_structured = parse_bom_csv(record.bom_csv)
    bom_text = bom_to_text(bom_structured) if bom_structured else None
    description = record.description or ""

    profile = infer_product_profile(description, bom_text)
    product_spec, extraction_evidence = extract_product_spec(
        description,
        bom_text,
        bom_structured=bom_structured,
        origin_country=record.origin_country,
        export_country=record.export_country,
        import_country=record.import_country,
    )
    compiled_facts_raw, fact_evidence = compile_facts_with_evidence(
        product_spec,
        description,
        bom_text,
        bom_structured=bom_structured,
        include_fact_evidence=True,
    )
    baseline_facts = baseline_facts_from_profile(profile)
    merged_facts = _merge_facts(baseline_facts, compiled_facts_raw)
    merged_facts, overrides_payload, override_evidence = _apply_fact_overrides(merged_facts, override_models)
    if override_evidence:
        fact_evidence = list(fact_evidence or [])
        fact_evidence.extend(override_evidence)
        fact_evidence.sort(
            key=lambda item: (
                item.fact_key,
                str(item.value),
                item.confidence,
                len(item.evidence),
            )
        )
    if (
        product_spec.product_category == ProductCategory.ELECTRONICS
        and attached_evidence_ids
        and merged_facts.get("electronics_insulation_documented") is None
    ):
        merged_facts["electronics_insulation_documented"] = True
        derived_from = [f"evidence:{eid}" for eid in attached_evidence_ids]
        doc_evidence = FactEvidenceModel(
            fact_key="electronics_insulation_documented",
            value=True,
            confidence=0.75,
            evidence=[],
            derived_from=derived_from,
            risk_reason="Declared insulated conductors via attached evidence",
        )
        if merged_facts.get("electronics_insulated_conductors") is None:
            merged_facts["electronics_insulated_conductors"] = True
            fact_evidence = list(fact_evidence or [])
            fact_evidence.append(
                FactEvidenceModel(
                    fact_key="electronics_insulated_conductors",
                    value=True,
                    confidence=0.7,
                    evidence=[],
                    derived_from=derived_from,
                    risk_reason="Documented insulation using attached evidence",
                )
            )
        fact_evidence = list(fact_evidence or [])
        fact_evidence.append(doc_evidence)

    normalized_facts, normalization_notes = normalize_compiled_facts(merged_facts)
    missing_inputs = validate_minimum_inputs(product_spec.model_dump(), normalized_facts)
    compiled_pack = {
        "raw": compiled_facts_raw,
        "baseline": baseline_facts,
        "normalized": normalized_facts,
    }
    compiled_pack.update(compiled_facts_raw)
    compiled_pack.update({key: value for key, value in merged_facts.items() if key not in compiled_pack})
    if overrides_payload:
        compiled_pack["overrides"] = overrides_payload
    if attached_evidence_ids:
        compiled_pack["attached_evidence_ids"] = attached_evidence_ids
    if session_id:
        compiled_pack["analysis_session_id"] = session_id

    if fact_evidence:
        fact_evidence = sorted(
            fact_evidence,
            key=lambda item: (
                item.fact_key,
                str(item.value),
                item.confidence,
                len(item.evidence),
            ),
        )

    resolved_context = law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    law_context_id = context_meta["context_id"]
    duty_rates = context_meta.get("duty_rates") or DUTY_RATES

    baseline_candidates, baseline_eval = _baseline_candidates(normalized_facts, atoms, duty_rates)
    baseline_assignment = _select_baseline_assignment(baseline_eval, baseline_candidates, duty_rates)
    requirement_registry = FactRequirementRegistry()
    baseline_duty_atoms = baseline_eval.duty_atoms
    baseline_duty = baseline_assignment.duty_rate if baseline_assignment.duty_rate is not None else (
        baseline_eval.duty_rate if baseline_eval.duty_rate is not None else (baseline_candidates[0].duty_rate if baseline_candidates else None)
    )
    baseline_confidence = baseline_assignment.confidence if baseline_assignment.confidence is not None else (
        baseline_candidates[0].confidence if baseline_candidates else 0.3
    )

    evidence_pack = _evidence_pack(
        product_spec=product_spec.model_dump(),
        compiled_facts=compiled_pack,
        fact_evidence=[item.model_dump() for item in fact_evidence] if fact_evidence else [],
        extraction_evidence=[item.model_dump() for item in extraction_evidence] if extraction_evidence else [],
        sku_metadata=build_sku_metadata_snapshot(
            sku_id=record.sku_id,
            description=description,
            bom_json=record.bom_json,
            bom_csv=record.bom_csv,
            origin_country=record.origin_country,
            export_country=record.export_country,
            import_country=record.import_country,
            declared_value_per_unit=record.declared_value_per_unit,
            annual_volume=record.annual_volume,
            metadata=record.metadata,
        ),
        analysis_session_id=session_id,
        attached_evidence_ids=attached_evidence_ids,
        fact_overrides=overrides_payload,
    )

    lever_requirements = _lever_requirement_gaps(product_spec, normalized_facts)
    conditional_pathways = _conditional_pathways(
        baseline_candidates,
        baseline_assignment,
        requirement_registry=requirement_registry,
    )
    questions = generate_missing_fact_questions(
        law_context=law_context_id,
        atoms=atoms,
        compiled_facts=normalized_facts,
        candidates=baseline_candidates,
        lever_requirements=lever_requirements,
    )
    blocking_reasons: list[str] = []
    for item in questions.questions:
        if not item.blocks_optimization:
            continue
        impacted: list[str] = []
        if item.candidate_rules_affected:
            impacted.append(f"atoms {', '.join(item.candidate_rules_affected)}")
        if item.lever_ids_affected:
            impacted.append(f"levers {', '.join(item.lever_ids_affected)}")
        impacted_text = " and ".join(impacted) if impacted else "optimization reasoning"
        blocking_reasons.append(f"blocks optimization because {item.fact_key} is required by {impacted_text}")
    has_blocking_questions = any(question.blocks_optimization for question in questions.questions)
    data_quality = {
        "missing_facts_remaining": len(questions.missing_facts),
        "evidence_attached_count": len(attached_evidence_ids),
        "overrides_applied_count": len(overrides_payload),
    }

    status: str = "OK_BASELINE_ONLY"
    why_not_optimized: list[str] = blocking_reasons + normalization_notes[:]
    suggestion_items: list[TariffSuggestionItemModel] = []
    proof_id: str | None = None
    proof_payload_hash: str | None = None

    if missing_inputs:
        status = "OK_BASELINE_ONLY"
        why_not_optimized = missing_inputs + why_not_optimized
    elif not baseline_eval.sat:
        status = "INSUFFICIENT_RULE_COVERAGE"
    elif has_blocking_questions and optimize:
        status = "OK_BASELINE_ONLY"
        if not blocking_reasons:
            blocking_reasons.append("blocks optimization because baseline candidates require missing facts")
        why_not_optimized = blocking_reasons + normalization_notes[:]
    elif optimize:
        levers = applicable_levers(spec=product_spec, facts=normalized_facts)
        for lever in levers:
            mutated = deepcopy(normalized_facts)
            mutated.update(lever.mutation)
            mutated_normalized, _ = normalize_compiled_facts(mutated)
            optimized_eval = _evaluate_scenario(atoms, mutated_normalized, duty_rates)
            optimized_rate = optimized_eval.duty_rate
            if not optimized_eval.sat or optimized_rate is None:
                continue
            optimized_duty_atoms = optimized_eval.duty_atoms

            optimized_candidates = generate_baseline_candidates(mutated_normalized, atoms, duty_rates, max_candidates=1)
            optimized_confidence = optimized_candidates[0].confidence if optimized_candidates else baseline_confidence

            savings_rate, savings_value, annual_value = _compute_savings(
                baseline_rate=baseline_duty,
                optimized_rate=optimized_rate,
                declared_value_per_unit=record.declared_value_per_unit or 100.0,
                annual_volume=record.annual_volume,
            )
            confidence_gain = optimized_confidence > baseline_confidence + 0.2
            if savings_rate <= 0 and not confidence_gain:
                continue

            risk = assess_tariff_risk(
                baseline_facts=normalized_facts,
                optimized_facts=mutated_normalized,
                baseline_active_atoms=baseline_eval.active_atoms,
                optimized_active_atoms=optimized_eval.active_atoms,
                baseline_duty_rate=baseline_duty or optimized_rate,
                optimized_duty_rate=optimized_rate,
            )
            adjusted_risk_score = max(risk.risk_score, lever.risk_floor)
            adjusted_grade = _defensibility_grade(adjusted_risk_score)

            proof_handle = _record_proof(
                law_context=law_context_id,
                context_meta=context_meta,
                baseline_facts=normalized_facts,
                optimized_facts=mutated_normalized,
                baseline_eval=baseline_eval,
                optimized_eval=optimized_eval,
                lever=lever,
                evidence_pack=evidence_pack,
            )

            suggestion_items.append(
                TariffSuggestionItemModel(
                    human_summary=lever.rationale,
                    lever_id=lever.lever_id,
                    lever_feasibility=lever.feasibility,
                    evidence_requirements=list(lever.evidence_requirements),
                    baseline_duty_rate=baseline_duty or optimized_rate,
                    optimized_duty_rate=optimized_rate,
                    savings_per_unit_rate=savings_rate,
                    savings_per_unit_value=savings_value,
                    annual_savings_value=annual_value,
                    best_mutation=dict(lever.mutation),
                    classification_confidence=optimized_confidence,
                    active_codes_baseline=sorted({atom.source_id for atom in baseline_duty_atoms}),
                    active_codes_optimized=sorted({atom.source_id for atom in optimized_duty_atoms}),
                    provenance_chain=_combined_provenance(baseline_duty_atoms, optimized_duty_atoms),
                    law_context=law_context_id,
                    proof_id=proof_handle.proof_id,
                    proof_payload_hash=proof_handle.proof_payload_hash,
                    risk_score=adjusted_risk_score,
                    defensibility_grade=adjusted_grade,
                    risk_reasons=risk.risk_reasons,
                    tariff_manifest_hash=context_meta["manifest_hash"],
                )
            )

        indexed_items = list(enumerate(suggestion_items))
        indexed_items.sort(key=lambda pair: _sort_key(pair[1], record.annual_volume, pair[0]))
        suggestion_items = [item for _, item in indexed_items]

        if suggestion_items:
            status = "OK_OPTIMIZED"
            proof_id = suggestion_items[0].proof_id
            proof_payload_hash = suggestion_items[0].proof_payload_hash
        else:
            status = "OK_BASELINE_ONLY"
            why_not_optimized.append("no feasible optimization identified")

    if proof_id is None:
        baseline_handle = record_tariff_proof(
            law_context=law_context_id,
            base_facts=normalized_facts,
            mutations={},
            baseline_active=baseline_eval.active_atoms,
            optimized_active=baseline_eval.active_atoms,
            baseline_sat=baseline_eval.sat,
            optimized_sat=baseline_eval.sat,
            baseline_duty_rate=baseline_duty,
            optimized_duty_rate=baseline_duty,
            baseline_duty_status=baseline_eval.duty_status,
            optimized_duty_status=baseline_eval.duty_status,
            baseline_scenario=normalized_facts,
            optimized_scenario=normalized_facts,
            baseline_unsat_core=baseline_eval.unsat_core,
            optimized_unsat_core=baseline_eval.unsat_core,
            provenance_chain=_provenance_for_atoms(baseline_duty_atoms, "baseline"),
            evidence_pack=evidence_pack,
            tariff_manifest_hash=context_meta["manifest_hash"],
        )
        proof_id = baseline_handle.proof_id
        proof_payload_hash = baseline_handle.proof_payload_hash

    best_optimization = suggestion_items[0] if suggestion_items else None
    optimized_section = SKUDossierOptimizedModel(suggestion=best_optimization) if best_optimization else None

    return TariffSkuDossierV2Model(
        status=status,
        sku_id=record.sku_id,
        law_context=law_context_id,
        tariff_manifest_hash=context_meta["manifest_hash"],
        proof_id=proof_id,
        proof_payload_hash=proof_payload_hash,
        baseline=SKUDossierBaselineModel(duty_rate=baseline_duty, candidates=baseline_candidates),
        optimized=optimized_section,
        baseline_assigned=baseline_assignment,
        conditional_pathways=conditional_pathways,
        questions=questions,
        product_spec=product_spec.model_dump(),
        compiled_facts=compiled_pack,
        fact_evidence=fact_evidence if evidence_requested else None,
        evidence_requested=evidence_requested,
        analysis_session_id=session_id,
        attached_evidence_ids=attached_evidence_ids,
        fact_overrides=overrides_payload or None,
        data_quality=data_quality,
        why_not_optimized=why_not_optimized,
        errors=None if status != "ERROR" else ["Unexpected error"],
    )
