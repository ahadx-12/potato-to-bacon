from __future__ import annotations

import json
from typing import Any, Dict, List

from potatobacon.tariff.engine import _build_provenance, run_tariff_hack
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.mutation_generator import (
    baseline_facts_from_profile,
    generate_candidate_mutations,
    human_summary_for_mutation,
    infer_product_profile,
)
from potatobacon.tariff.models import (
    BaselineCandidateModel,
    TariffSuggestRequestModel,
    TariffSuggestResponseModel,
    TariffSuggestionItemModel,
)
from potatobacon.tariff.normalizer import normalize_compiled_facts, validate_minimum_inputs
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
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
    candidates = generate_candidate_mutations(profile)
    compiled_pack = {"raw": compiled_facts, "normalized": normalized_facts}
    compiled_pack.update(compiled_facts)

    resolved_context = request.law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    law_context = context_meta["context_id"]

    declared_value = request.declared_value_per_unit or 100.0
    seed = request.seed or 2025

    baseline_candidates: List[BaselineCandidateModel] = []
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
            generated_candidates_count=len(candidates),
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            why_not_optimized=missing_inputs + why_not_optimized,
            proof_id=None,
            proof_payload_hash=None,
        )

    baseline_sat, baseline_active_atoms, baseline_unsat = analyze_scenario(normalized_facts, atoms)
    duty_atoms = [atom for atom in baseline_active_atoms if atom.source_id in DUTY_RATES]
    if baseline_sat and duty_atoms:
        active_atom = duty_atoms[-1]
        baseline_candidates.append(
            BaselineCandidateModel(
                candidate_id=active_atom.source_id,
                active_codes=sorted({atom.source_id for atom in duty_atoms}),
                duty_rate=float(DUTY_RATES.get(active_atom.source_id, 0.0)),
                provenance_chain=_build_provenance(duty_atoms, "baseline"),
                confidence=0.8,
                missing_facts=[],
                compliance_flags={"requires_ruling_review": bool(baseline_unsat)},
            )
        )
    elif baseline_sat:
        baseline_candidates.append(
            BaselineCandidateModel(
                candidate_id="NO_DUTY_RULE",
                active_codes=[],
                duty_rate=0.0,
                provenance_chain=[],
                confidence=0.3,
                missing_facts=[],
                compliance_flags={"requires_rule_extension": True},
            )
        )
        why_not_optimized.append("no duty rule activated for baseline scenario")
        return TariffSuggestResponseModel(
            status="OK_BASELINE_ONLY",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=len(candidates),
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            why_not_optimized=why_not_optimized,
            proof_id=None,
            proof_payload_hash=None,
        )
    else:
        return TariffSuggestResponseModel(
            status="INSUFFICIENT_RULE_COVERAGE",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=len(candidates),
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            why_not_optimized=why_not_optimized,
            proof_id=None,
            proof_payload_hash=None,
        )

    if not candidates:
        why_not_optimized.append("no mutation templates available for profile")
        baseline_only = run_tariff_hack(
            base_facts=normalized_facts,
            mutations=None,
            law_context=law_context,
            seed=seed,
            evidence_pack={
                "product_spec": spec.model_dump(),
                "compiled_facts": compiled_pack,
                "fact_evidence": [item.model_dump() for item in fact_evidence],
                "extraction_evidence": [item.model_dump() for item in extraction_evidence],
            },
        )
        baseline_candidates = [
            BaselineCandidateModel(
                candidate_id=baseline_only.active_codes_baseline[-1] if baseline_only.active_codes_baseline else "baseline",
                active_codes=baseline_only.active_codes_baseline,
                duty_rate=baseline_only.baseline_duty_rate,
                provenance_chain=baseline_only.provenance_chain,
                confidence=0.8,
                missing_facts=[],
                compliance_flags={"requires_ruling_review": False},
            )
        ]
        return TariffSuggestResponseModel(
            status="OK_BASELINE_ONLY",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=len(candidates),
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            why_not_optimized=why_not_optimized,
            proof_id=baseline_only.proof_id,
            proof_payload_hash=baseline_only.proof_payload_hash,
        )

    for mutation in candidates:
        dossier = run_tariff_hack(
            base_facts=normalized_facts,
            mutations=mutation,
            law_context=law_context,
            seed=seed,
            evidence_pack={
                "product_spec": spec.model_dump(),
                "compiled_facts": compiled_pack,
                "fact_evidence": [item.model_dump() for item in fact_evidence],
                "extraction_evidence": [item.model_dump() for item in extraction_evidence],
            },
        )
        proof_id = dossier.proof_id
        proof_payload_hash = dossier.proof_payload_hash

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

    improving = [item for item in suggestion_items if item.optimized_duty_rate < item.baseline_duty_rate]
    ordered_suggestions: List[TariffSuggestionItemModel] = []
    response_law_context = law_context
    status: str

    if improving:
        indexed_items = list(enumerate(improving))
        indexed_items.sort(key=lambda pair: _sort_key(pair[1], request.annual_volume, pair[0]))
        top_k = request.top_k or 5
        ordered_suggestions = [item for _, item in indexed_items[:top_k]]
        if ordered_suggestions and ordered_suggestions[0].law_context:
            response_law_context = ordered_suggestions[0].law_context
        status = "OK_OPTIMIZED"
        proof_id = ordered_suggestions[0].proof_id
        proof_payload_hash = ordered_suggestions[0].proof_payload_hash
    else:
        status = "OK_BASELINE_ONLY"
        why_not_optimized.append("no feasible optimization identified")
        baseline_only = run_tariff_hack(
            base_facts=normalized_facts,
            mutations=None,
            law_context=law_context,
            seed=seed,
            evidence_pack={
                "product_spec": spec.model_dump(),
                "compiled_facts": compiled_pack,
                "fact_evidence": [item.model_dump() for item in fact_evidence],
                "extraction_evidence": [item.model_dump() for item in extraction_evidence],
            },
        )
        baseline_candidates = [
            BaselineCandidateModel(
                candidate_id=baseline_only.active_codes_baseline[-1] if baseline_only.active_codes_baseline else "baseline",
                active_codes=baseline_only.active_codes_baseline,
                duty_rate=baseline_only.baseline_duty_rate,
                provenance_chain=baseline_only.provenance_chain,
                confidence=0.8,
                missing_facts=[],
                compliance_flags={"requires_ruling_review": False},
            )
        ]
        proof_id = baseline_only.proof_id
        proof_payload_hash = baseline_only.proof_payload_hash

    return TariffSuggestResponseModel(
        status=status,
        sku_id=request.sku_id,
        description=request.description,
        law_context=response_law_context,
        baseline_scenario=normalized_facts,
        generated_candidates_count=len(candidates),
        suggestions=ordered_suggestions,
        tariff_manifest_hash=context_meta["manifest_hash"],
        fact_evidence=fact_evidence if request.include_fact_evidence else None,
        product_spec=spec if request.include_fact_evidence else None,
        baseline_candidates=baseline_candidates,
        why_not_optimized=why_not_optimized,
        proof_id=proof_id,
        proof_payload_hash=proof_payload_hash,
    )
