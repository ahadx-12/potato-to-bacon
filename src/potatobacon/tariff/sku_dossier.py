from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
from potatobacon.tariff.candidate_search import generate_baseline_candidates
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.levers import applicable_levers
from potatobacon.tariff.models import BaselineCandidateModel, FactEvidenceModel, TariffSuggestionItemModel
from potatobacon.tariff.mutation_generator import baseline_facts_from_profile, infer_product_profile
from potatobacon.tariff.normalizer import normalize_compiled_facts, validate_minimum_inputs
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.questions import generate_missing_fact_questions
from potatobacon.tariff.risk import assess_tariff_risk
from potatobacon.tariff.sku_models import (
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


def _baseline_candidates(
    normalized_facts: Dict[str, Any], atoms: Any
) -> tuple[list[BaselineCandidateModel], Any]:
    baseline_candidates = generate_baseline_candidates(normalized_facts, atoms, DUTY_RATES, max_candidates=5)
    baseline_eval = _evaluate_scenario(atoms, normalized_facts)
    return baseline_candidates, baseline_eval


def _provenance_for_atoms(atoms: Any, scenario_label: str) -> list[Dict[str, Any]]:
    chain: list[Dict[str, Any]] = []
    for atom in atoms or []:
        chain.append(
            {
                "scenario": scenario_label,
                "source_id": atom.source_id,
                "statute": getattr(atom, "statute", ""),
                "section": getattr(atom, "section", ""),
                "text": getattr(atom, "text", ""),
                "jurisdiction": atom.outcome.get("jurisdiction", ""),
            }
        )
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
    normalized_facts, normalization_notes = normalize_compiled_facts(merged_facts)
    missing_inputs = validate_minimum_inputs(product_spec.model_dump(), normalized_facts)
    compiled_pack = {
        "raw": compiled_facts_raw,
        "baseline": baseline_facts,
        "normalized": normalized_facts,
    }
    compiled_pack.update(compiled_facts_raw)
    if overrides_payload:
        compiled_pack["overrides"] = overrides_payload
    if attached_evidence_ids:
        compiled_pack["attached_evidence_ids"] = attached_evidence_ids
    if session_id:
        compiled_pack["analysis_session_id"] = session_id

    resolved_context = law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    law_context_id = context_meta["context_id"]

    baseline_candidates, baseline_eval = _baseline_candidates(normalized_facts, atoms)
    baseline_duty_atoms = baseline_eval.duty_atoms
    baseline_duty = baseline_eval.duty_rate if baseline_eval.duty_rate is not None else (
        baseline_candidates[0].duty_rate if baseline_candidates else None
    )
    baseline_confidence = baseline_candidates[0].confidence if baseline_candidates else 0.3

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

    questions = generate_missing_fact_questions(
        law_context=law_context_id,
        atoms=atoms,
        compiled_facts=normalized_facts,
        candidates=baseline_candidates,
    )
    active_ids = {atom.source_id for atom in baseline_duty_atoms}
    blocking_missing = sorted(
        {
            item.fact_key
            for item in questions.questions
            if item.candidate_rules_affected
            and any(rule.split(":")[0] in active_ids for rule in item.candidate_rules_affected)
        }
    )
    has_blocking_questions = bool(questions.missing_facts)
    blocking_for_status = blocking_missing or questions.missing_facts
    data_quality = {
        "fact_overrides_applied": len(overrides_payload),
        "attached_evidence_references": len(attached_evidence_ids),
        "missing_facts_remaining": len(questions.missing_facts),
    }

    status: str = "OK_BASELINE_ONLY"
    why_not_optimized: list[str] = normalization_notes[:]
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
        why_not_optimized = blocking_for_status + why_not_optimized
    elif optimize:
        levers = applicable_levers(spec=product_spec, facts=normalized_facts)
        for lever in levers:
            mutated = deepcopy(normalized_facts)
            mutated.update(lever.mutation)
            mutated_normalized, _ = normalize_compiled_facts(mutated)
            optimized_eval = _evaluate_scenario(atoms, mutated_normalized)
            optimized_rate = optimized_eval.duty_rate
            if not optimized_eval.sat or optimized_rate is None:
                continue
            optimized_duty_atoms = optimized_eval.duty_atoms

            optimized_candidates = generate_baseline_candidates(mutated_normalized, atoms, DUTY_RATES, max_candidates=1)
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
