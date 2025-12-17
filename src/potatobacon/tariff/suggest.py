from __future__ import annotations

import json
from typing import Any, Dict, List

from potatobacon.tariff.engine import run_tariff_hack
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.mutation_generator import (
    baseline_facts_from_profile,
    generate_candidate_mutations,
    human_summary_for_mutation,
    infer_product_profile,
)
from potatobacon.tariff.models import (
    TariffSuggestRequestModel,
    TariffSuggestResponseModel,
    TariffSuggestionItemModel,
)
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.risk import assess_tariff_risk
from potatobacon.law.solver_z3 import analyze_scenario


def _compute_savings(
    baseline_rate: float,
    optimized_rate: float,
    declared_value_per_unit: float,
    annual_volume: int | None,
) -> tuple[float, float, float | None]:
    rate_delta = baseline_rate - optimized_rate
    savings_per_unit_value = rate_delta / 100.0 * declared_value_per_unit
    annual_savings_value = None
    if annual_volume is not None:
        annual_savings_value = savings_per_unit_value * annual_volume
    return rate_delta, savings_per_unit_value, annual_savings_value


def _sort_key(item: TariffSuggestionItemModel, annual_volume: int | None, index: int) -> tuple:
    primary = item.annual_savings_value if annual_volume is not None else item.savings_per_unit_value
    primary_value = primary if primary is not None else 0.0
    return (-primary_value, item.human_summary, index)


def suggest_tariff_optimizations(
    request: TariffSuggestRequestModel,
) -> TariffSuggestResponseModel:
    """Generate, evaluate, and rank tariff optimization suggestions."""

    profile = infer_product_profile(request.description, request.bom_text)
    spec, extraction_evidence = extract_product_spec(request.description, request.bom_text)
    compiled_facts, fact_evidence = compile_facts_with_evidence(
        spec, request.description, request.bom_text
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
    candidates = generate_candidate_mutations(profile)

    resolved_context = request.law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    law_context = context_meta["context_id"]

    if not baseline_facts or not candidates:
        return TariffSuggestResponseModel(
            status="NO_CANDIDATES",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=baseline_facts,
            generated_candidates_count=len(candidates),
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
        )

    declared_value = request.declared_value_per_unit or 100.0
    seed = request.seed or 2025

    suggestion_items: List[TariffSuggestionItemModel] = []

    for mutation in candidates:
        dossier = run_tariff_hack(
            base_facts=baseline_facts,
            mutations=mutation,
            law_context=law_context,
            seed=seed,
            evidence_pack={
                "product_spec": spec.model_dump(),
                "compiled_facts": compiled_facts,
                "fact_evidence": [item.model_dump() for item in fact_evidence],
                "extraction_evidence": [item.model_dump() for item in extraction_evidence],
            },
        )

        savings_rate, savings_value, annual_value = _compute_savings(
            baseline_rate=dossier.baseline_duty_rate,
            optimized_rate=dossier.optimized_duty_rate,
            declared_value_per_unit=declared_value,
            annual_volume=request.annual_volume,
        )

        _, baseline_active_atoms, _ = analyze_scenario(dossier.baseline_scenario, atoms)
        _, optimized_active_atoms, _ = analyze_scenario(dossier.optimized_scenario, atoms)

        risk = assess_tariff_risk(
            baseline_facts=dossier.baseline_scenario,
            optimized_facts=dossier.optimized_scenario,
            baseline_active_atoms=baseline_active_atoms,
            optimized_active_atoms=optimized_active_atoms,
            baseline_duty_rate=dossier.baseline_duty_rate,
            optimized_duty_rate=dossier.optimized_duty_rate,
        )

        suggestion_items.append(
            TariffSuggestionItemModel(
                human_summary=human_summary_for_mutation(profile, mutation),
                baseline_duty_rate=dossier.baseline_duty_rate,
                optimized_duty_rate=dossier.optimized_duty_rate,
                savings_per_unit_rate=savings_rate,
                savings_per_unit_value=savings_value,
                annual_savings_value=annual_value,
                best_mutation=mutation,
                active_codes_baseline=dossier.active_codes_baseline,
                active_codes_optimized=dossier.active_codes_optimized,
                provenance_chain=dossier.provenance_chain,
                law_context=dossier.law_context,
                proof_id=dossier.proof_id,
                proof_payload_hash=dossier.proof_payload_hash,
                risk_score=risk.risk_score,
                defensibility_grade=risk.defensibility_grade,
                risk_reasons=risk.risk_reasons,
                tariff_manifest_hash=dossier.tariff_manifest_hash,
            )
        )

    indexed_items = list(enumerate(suggestion_items))
    indexed_items.sort(key=lambda pair: _sort_key(pair[1], request.annual_volume, pair[0]))

    top_k = request.top_k or 5
    ordered_suggestions = [item for _, item in indexed_items[:top_k]]

    response_law_context = law_context
    if ordered_suggestions and ordered_suggestions[0].law_context:
        response_law_context = ordered_suggestions[0].law_context

    status: str = "OK" if ordered_suggestions else "NO_CANDIDATES"

    return TariffSuggestResponseModel(
        status=status,
        sku_id=request.sku_id,
        description=request.description,
        law_context=response_law_context,
        baseline_scenario=baseline_facts,
        generated_candidates_count=len(candidates),
        suggestions=ordered_suggestions,
        tariff_manifest_hash=context_meta["manifest_hash"],
        fact_evidence=fact_evidence if request.include_fact_evidence else None,
        product_spec=spec if request.include_fact_evidence else None,
    )
