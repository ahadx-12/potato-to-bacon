from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence

from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
from potatobacon.tariff.candidate_search import generate_baseline_candidates
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.engine import _build_provenance
from potatobacon.tariff.levers import LeverModel, applicable_levers
from potatobacon.tariff.mutation_generator import baseline_facts_from_profile, infer_product_profile
from potatobacon.tariff.models import (
    BaselineCandidateModel,
    TariffSuggestRequestModel,
    TariffSuggestResponseModel,
    TariffSuggestionItemModel,
)
from potatobacon.tariff.sku_models import build_sku_metadata_snapshot
from potatobacon.tariff.normalizer import normalize_compiled_facts, validate_minimum_inputs
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.risk import assess_tariff_risk
from potatobacon.law.solver_z3 import PolicyAtom, analyze_scenario


def _compute_savings(
    baseline_rate: float | None,
    optimized_rate: float,
    declared_value_per_unit: float,
    annual_volume: int | None,
) -> tuple[float, float, float | None]:
    base = baseline_rate if baseline_rate is not None else optimized_rate
    rate_delta = base - optimized_rate
    savings_per_unit_value = rate_delta / 100.0 * declared_value_per_unit
    annual_savings_value = None
    if annual_volume is not None:
        annual_savings_value = savings_per_unit_value * annual_volume
    return rate_delta, savings_per_unit_value, annual_savings_value


def _defensibility_grade(score: int) -> str:
    if score < 30:
        return "A"
    if score < 60:
        return "B"
    return "C"


def _duty_atoms(active_atoms: Sequence[PolicyAtom]) -> List[PolicyAtom]:
    return [atom for atom in active_atoms if atom.source_id in DUTY_RATES]


def _evaluate_scenario(atoms: Sequence[PolicyAtom], facts: Mapping[str, object]):
    sat, active_atoms, unsat_core = analyze_scenario(facts, atoms)
    duty_atoms = _duty_atoms(active_atoms)
    duty_rate = None
    if sat and duty_atoms:
        ranked = sorted(
            duty_atoms,
            key=lambda atom: (
                float(DUTY_RATES[atom.source_id]),
                -len(atom.guard),
                atom.source_id,
            ),
        )
        duty_atoms = ranked
        duty_rate = float(DUTY_RATES[ranked[0].source_id])
    return sat, active_atoms, unsat_core, duty_atoms, duty_rate


def _sort_key(item: TariffSuggestionItemModel, annual_volume: int | None, index: int) -> tuple:
    primary = item.annual_savings_value if annual_volume is not None else item.savings_per_unit_value
    primary_value = primary if primary is not None else item.savings_per_unit_rate
    risk_score = item.risk_score if item.risk_score is not None else 999
    confidence = item.classification_confidence if item.classification_confidence is not None else 0.0
    return (-primary_value, risk_score, -confidence, item.lever_id or "", index)


def _record_proof(
    *,
    law_context: str,
    context_meta: Mapping[str, Any],
    baseline_facts: Mapping[str, object],
    optimized_facts: Mapping[str, object],
    baseline_eval: tuple,
    optimized_eval: tuple,
    lever: LeverModel | None,
    evidence_pack: Mapping[str, Any] | None = None,
):
    _, baseline_active, baseline_unsat, _, baseline_rate = baseline_eval
    _, optimized_active, optimized_unsat, _, optimized_rate = optimized_eval
    provenance_chain: List[Dict[str, Any]] = []
    provenance_chain.extend(_build_provenance(_duty_atoms(baseline_active), "baseline"))
    provenance_chain.extend(_build_provenance(_duty_atoms(optimized_active), "optimized"))
    provenance_chain.sort(
        key=lambda item: (
            item.get("source_id", ""),
            item.get("section", ""),
            item.get("text", ""),
            item.get("scenario", ""),
        )
    )

    return record_tariff_proof(
        law_context=law_context,
        base_facts=dict(baseline_facts),
        mutations=lever.mutation if lever else {},
        baseline_active=baseline_active,
        optimized_active=optimized_active,
        baseline_sat=baseline_eval[0],
        optimized_sat=optimized_eval[0],
        baseline_duty_rate=baseline_rate,
        optimized_duty_rate=optimized_rate,
        baseline_scenario=dict(baseline_facts),
        optimized_scenario=dict(optimized_facts),
        baseline_unsat_core=baseline_unsat,
        optimized_unsat_core=optimized_unsat,
        provenance_chain=provenance_chain,
        evidence_pack=evidence_pack,
        tariff_manifest_hash=context_meta.get("manifest_hash"),
    )


def suggest_tariff_optimizations(
    request: TariffSuggestRequestModel,
) -> TariffSuggestResponseModel:
    """Generate, evaluate, and rank tariff optimization suggestions."""

    bom_structured = request.bom_json
    if bom_structured is None and request.bom_csv:
        bom_structured = parse_bom_csv(request.bom_csv)

    normalized_bom_text = request.bom_text
    if bom_structured is not None:
        normalized_bom_text = bom_to_text(bom_structured)

    profile = infer_product_profile(request.description, normalized_bom_text)
    spec, extraction_evidence = extract_product_spec(
        request.description,
        normalized_bom_text,
        bom_structured=bom_structured,
        origin_country=request.origin_country,
        export_country=request.export_country,
        import_country=request.import_country,
    )
    compiled_facts, fact_evidence = compile_facts_with_evidence(
        spec,
        request.description,
        normalized_bom_text,
        bom_structured=bom_structured,
        include_fact_evidence=request.include_fact_evidence,
    )
    if fact_evidence:
        fact_evidence = sorted(
            fact_evidence,
            key=lambda item: (
                item.fact_key,
                json.dumps(item.value, sort_keys=True),
                len(item.evidence),
            ),
        )
    if extraction_evidence:
        extraction_evidence = sorted(
            extraction_evidence,
            key=lambda item: (item.source, item.start or -1, item.end or -1, item.snippet),
        )

    baseline_facts = baseline_facts_from_profile(profile)
    if compiled_facts:
        for key, value in compiled_facts.items():
            if value or key not in baseline_facts:
                baseline_facts[key] = value

    normalized_facts, normalization_notes = normalize_compiled_facts(baseline_facts)
    missing_inputs = validate_minimum_inputs(spec.model_dump(), normalized_facts)
    compiled_pack = {"raw": compiled_facts, "normalized": normalized_facts}
    compiled_pack.update(compiled_facts)

    resolved_context = request.law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    law_context = context_meta["context_id"]

    declared_value = request.declared_value_per_unit or 100.0
    seed = request.seed or 2025  # reserved for future stochastic flows

    baseline_candidates = generate_baseline_candidates(normalized_facts, atoms, DUTY_RATES, max_candidates=5)
    baseline_eval = _evaluate_scenario(atoms, normalized_facts)
    baseline_rate = baseline_candidates[0].duty_rate if baseline_candidates else baseline_eval[4]
    baseline_confidence = baseline_candidates[0].confidence if baseline_candidates else 0.3

    suggestion_items: List[TariffSuggestionItemModel] = []
    why_not_optimized: List[str] = normalization_notes[:]
    proof_id: str | None = None
    proof_payload_hash: str | None = None

    if missing_inputs:
        return TariffSuggestResponseModel(
            status="INSUFFICIENT_INPUTS",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=0,
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            why_not_optimized=missing_inputs + why_not_optimized,
            proof_id=None,
            proof_payload_hash=None,
        )

    if not baseline_eval[0]:
        return TariffSuggestResponseModel(
            status="INSUFFICIENT_RULE_COVERAGE",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=0,
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            why_not_optimized=why_not_optimized,
            proof_id=None,
            proof_payload_hash=None,
        )

    levers = applicable_levers(spec=spec, facts=normalized_facts)
    evidence_pack = {
        "product_spec": spec.model_dump(),
        "compiled_facts": compiled_pack,
        "fact_evidence": [item.model_dump() for item in fact_evidence] if fact_evidence else [],
        "extraction_evidence": [item.model_dump() for item in extraction_evidence] if extraction_evidence else [],
    }
    sku_metadata = build_sku_metadata_snapshot(
        sku_id=request.sku_id,
        description=request.description,
        bom_json=request.bom_json,
        bom_csv=request.bom_csv,
        origin_country=request.origin_country,
        export_country=request.export_country,
        import_country=request.import_country,
        declared_value_per_unit=request.declared_value_per_unit,
        annual_volume=request.annual_volume,
    )
    if sku_metadata:
        evidence_pack["sku_metadata"] = sku_metadata

    for lever in levers:
        mutated = deepcopy(normalized_facts)
        mutated.update(lever.mutation)
        mutated_normalized, _ = normalize_compiled_facts(mutated)

        optimized_eval = _evaluate_scenario(atoms, mutated_normalized)
        optimized_sat, _, _, optimized_duty_atoms, optimized_rate = optimized_eval
        if not optimized_sat or optimized_rate is None:
            continue

        optimized_candidates = generate_baseline_candidates(mutated_normalized, atoms, DUTY_RATES, max_candidates=1)
        optimized_confidence = optimized_candidates[0].confidence if optimized_candidates else baseline_confidence

        savings_rate, savings_value, annual_value = _compute_savings(
            baseline_rate=baseline_rate,
            optimized_rate=optimized_rate,
            declared_value_per_unit=declared_value,
            annual_volume=request.annual_volume,
        )

        confidence_gain = optimized_confidence > baseline_confidence + 0.2
        if savings_rate <= 0 and not confidence_gain:
            continue

        _, baseline_active_atoms, baseline_unsat, baseline_duty_atoms, _ = baseline_eval
        risk = assess_tariff_risk(
            baseline_facts=normalized_facts,
            optimized_facts=mutated_normalized,
            baseline_active_atoms=baseline_active_atoms,
            optimized_active_atoms=optimized_eval[1],
            baseline_duty_rate=baseline_rate or optimized_rate,
            optimized_duty_rate=optimized_rate,
        )
        adjusted_risk_score = max(risk.risk_score, lever.risk_floor)
        adjusted_grade = _defensibility_grade(adjusted_risk_score)

        proof_handle = _record_proof(
            law_context=law_context,
            context_meta=context_meta,
            baseline_facts=normalized_facts,
            optimized_facts=mutated_normalized,
            baseline_eval=baseline_eval,
            optimized_eval=optimized_eval,
            lever=lever,
            evidence_pack=evidence_pack,
        )

        provenance_chain: List[Dict[str, Any]] = []
        provenance_chain.extend(_build_provenance(baseline_duty_atoms, "baseline"))
        provenance_chain.extend(_build_provenance(optimized_duty_atoms, "optimized"))
        provenance_chain.sort(
            key=lambda item: (
                item.get("source_id", ""),
                item.get("section", ""),
                item.get("text", ""),
                item.get("scenario", ""),
            )
        )

        suggestion_items.append(
            TariffSuggestionItemModel(
                human_summary=lever.rationale,
                lever_id=lever.lever_id,
                lever_feasibility=lever.feasibility,
                evidence_requirements=list(lever.evidence_requirements),
                baseline_duty_rate=baseline_rate or optimized_rate,
                optimized_duty_rate=optimized_rate,
                savings_per_unit_rate=savings_rate,
                savings_per_unit_value=savings_value,
                annual_savings_value=annual_value,
                best_mutation=dict(lever.mutation),
                classification_confidence=optimized_confidence,
                active_codes_baseline=sorted({atom.source_id for atom in baseline_duty_atoms}),
                active_codes_optimized=sorted({atom.source_id for atom in optimized_duty_atoms}),
                provenance_chain=provenance_chain,
                law_context=law_context,
                proof_id=proof_handle.proof_id,
                proof_payload_hash=proof_handle.proof_payload_hash,
                risk_score=adjusted_risk_score,
                defensibility_grade=adjusted_grade,
                risk_reasons=risk.risk_reasons,
                tariff_manifest_hash=context_meta["manifest_hash"],
            )
        )

    indexed_items = list(enumerate(suggestion_items))
    indexed_items.sort(key=lambda pair: _sort_key(pair[1], request.annual_volume, pair[0]))
    top_k = request.top_k or 5
    ordered_suggestions = [item for _, item in indexed_items[:top_k]]

    status = "OK_OPTIMIZED" if ordered_suggestions else "OK_BASELINE_ONLY"
    if not ordered_suggestions:
        why_not_optimized.append("no feasible optimization identified")
        baseline_proof = _record_proof(
            law_context=law_context,
            context_meta=context_meta,
            baseline_facts=normalized_facts,
            optimized_facts=normalized_facts,
            baseline_eval=baseline_eval,
            optimized_eval=baseline_eval,
            lever=None,
            evidence_pack=evidence_pack,
        )
        proof_id = baseline_proof.proof_id
        proof_payload_hash = baseline_proof.proof_payload_hash
    else:
        proof_id = ordered_suggestions[0].proof_id
        proof_payload_hash = ordered_suggestions[0].proof_payload_hash

    return TariffSuggestResponseModel(
        status=status,
        sku_id=request.sku_id,
        description=request.description,
        law_context=law_context,
        baseline_scenario=normalized_facts,
        generated_candidates_count=len(levers),
        suggestions=ordered_suggestions,
        tariff_manifest_hash=context_meta["manifest_hash"],
        fact_evidence=fact_evidence if request.include_fact_evidence else None,
        product_spec=spec if request.include_fact_evidence else None,
        baseline_candidates=baseline_candidates,
        why_not_optimized=why_not_optimized,
        proof_id=proof_id,
        proof_payload_hash=proof_payload_hash,
    )
