from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.security import require_api_key
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
from potatobacon.tariff.models import (
    TariffSkuDossierModel,
    TariffSuggestRequestModel,
)
from potatobacon.tariff.mutation_generator import baseline_facts_from_profile, infer_product_profile
from potatobacon.tariff.normalizer import normalize_compiled_facts, validate_minimum_inputs
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.suggest import suggest_tariff_optimizations

router = APIRouter(
    prefix="/api",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/tariff/sku/dossier", response_model=TariffSkuDossierModel)
def build_tariff_sku_dossier(
    request: TariffSuggestRequestModel,
) -> TariffSkuDossierModel:
    bom_structured = request.bom_json
    if bom_structured is None and request.bom_csv:
        bom_structured = parse_bom_csv(request.bom_csv)
    normalized_bom_text = request.bom_text
    if bom_structured is not None:
        normalized_bom_text = bom_to_text(bom_structured)

    profile = infer_product_profile(request.description, normalized_bom_text)
    product_spec, _ = extract_product_spec(
        request.description,
        normalized_bom_text,
        bom_structured=bom_structured,
        origin_country=request.origin_country,
        export_country=request.export_country,
        import_country=request.import_country,
    )
    compiled_facts_raw, fact_evidence = compile_facts_with_evidence(
        product_spec,
        request.description,
        normalized_bom_text,
        bom_structured=bom_structured,
        include_fact_evidence=request.include_fact_evidence,
    )
    if fact_evidence:
        fact_evidence = sorted(
            fact_evidence,
            key=lambda item: (item.fact_key, str(item.value), item.confidence, len(item.evidence)),
        )

    baseline_facts = baseline_facts_from_profile(profile)
    merged_facts = dict(baseline_facts)
    merged_facts.update(compiled_facts_raw)
    normalized_facts, normalization_notes = normalize_compiled_facts(merged_facts)
    missing_inputs = validate_minimum_inputs(product_spec.model_dump(), normalized_facts)

    suggest_request = TariffSuggestRequestModel(
        **{**request.model_dump(), "bom_json": bom_structured, "bom_text": normalized_bom_text},
    )
    suggest_response = suggest_tariff_optimizations(suggest_request)

    best = suggest_response.suggestions[0] if suggest_response.suggestions else None
    best_optimization = best.model_dump() if best else None
    proof_id = best.proof_id if best else suggest_response.proof_id
    proof_payload_hash = best.proof_payload_hash if best else suggest_response.proof_payload_hash

    compiled_facts = {
        "raw": compiled_facts_raw,
        "baseline": baseline_facts,
        "normalized": normalized_facts,
    }

    return TariffSkuDossierModel(
        status=suggest_response.status,
        sku_id=request.sku_id,
        law_context=suggest_response.law_context or request.law_context or "",
        tariff_manifest_hash=suggest_response.tariff_manifest_hash or "",
        proof_id=proof_id,
        proof_payload_hash=proof_payload_hash,
        product_spec=product_spec.model_dump(),
        compiled_facts=compiled_facts,
        fact_evidence=[item.model_dump() for item in fact_evidence] if request.include_fact_evidence else None,
        baseline_candidates=suggest_response.baseline_candidates,
        best_optimization=best_optimization,
        why_not_optimized=suggest_response.why_not_optimized or normalization_notes or missing_inputs,
        errors=None if suggest_response.status != "ERROR" else ["Unexpected error"],
    )
