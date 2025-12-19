from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID
from potatobacon.tariff.models import (
    TariffBatchScanRequestModel,
    TariffBatchScanResponseModel,
    TariffBatchSkuResultModel,
    TariffSuggestRequestModel,
    TariffSuggestionItemModel,
)
from potatobacon.tariff.suggest import suggest_tariff_optimizations


def _score_suggestion(
    best: TariffSuggestionItemModel | None,
    risk_adjusted: bool,
    risk_penalty: float,
) -> float:
    if best is None:
        return 0.0
    if best.annual_savings_value is not None:
        base_score = best.annual_savings_value
    elif best.savings_per_unit_value is not None:
        base_score = best.savings_per_unit_value
    else:
        base_score = 0.0

    if not risk_adjusted:
        return base_score

    risk_score = best.risk_score or 0
    factor = 1.0 - (risk_penalty * (risk_score / 100.0))
    factor = max(0.0, factor)
    return base_score * factor


def batch_scan_tariffs(request: TariffBatchScanRequestModel) -> TariffBatchScanResponseModel:
    """Run tariff suggestion engine across multiple SKUs and rank the outcomes."""

    ranked_results: List[TariffBatchSkuResultModel] = []
    skipped: List[TariffBatchSkuResultModel] = []

    for sku in request.skus:
        try:
            context = sku.law_context or request.law_context or DEFAULT_CONTEXT_ID
            suggest_request = TariffSuggestRequestModel(
                sku_id=sku.sku_id,
                description=sku.description,
                bom_text=sku.bom_text,
                declared_value_per_unit=sku.declared_value_per_unit,
                annual_volume=sku.annual_volume,
                law_context=context,
                top_k=request.top_k_per_sku,
                seed=request.seed,
            )
            suggest_response = suggest_tariff_optimizations(suggest_request)

            if suggest_response.status in {
                "OK_BASELINE_ONLY",
                "INSUFFICIENT_RULE_COVERAGE",
                "INSUFFICIENT_INPUTS",
            } or not suggest_response.suggestions:
                suggestions = suggest_response.suggestions if request.include_all_suggestions else None
                skipped.append(
                    TariffBatchSkuResultModel(
                        sku_id=sku.sku_id,
                        description=sku.description,
                        status=suggest_response.status,
                        law_context=suggest_response.law_context,
                        tariff_manifest_hash=suggest_response.tariff_manifest_hash,
                        baseline_scenario=suggest_response.baseline_scenario,
                        best=None,
                        suggestions=suggestions,
                        rank_score=None,
                        error=None,
                    )
                )
                continue

            suggestions = suggest_response.suggestions
            best = suggestions[0]
            rank_score = _score_suggestion(
                best,
                risk_adjusted=request.risk_adjusted_ranking,
                risk_penalty=request.risk_penalty,
            )

            ranked_results.append(
                TariffBatchSkuResultModel(
                    sku_id=sku.sku_id,
                    description=sku.description,
                    status=suggest_response.status,
                    law_context=suggest_response.law_context,
                    tariff_manifest_hash=suggest_response.tariff_manifest_hash,
                    baseline_scenario=suggest_response.baseline_scenario,
                    best=best,
                    suggestions=suggestions if request.include_all_suggestions else None,
                    rank_score=rank_score,
                    error=None,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(exc, KeyError):
                raise
            skipped.append(
                TariffBatchSkuResultModel(
                    sku_id=sku.sku_id,
                    description=sku.description,
                    status="ERROR",
                    law_context=sku.law_context or request.law_context or DEFAULT_CONTEXT_ID,
                    baseline_scenario={},
                    best=None,
                    suggestions=None,
                    rank_score=None,
                    error=str(exc),
                )
            )

    ranked_results.sort(
        key=lambda res: (
            -_score_suggestion(
                res.best,
                risk_adjusted=request.risk_adjusted_ranking,
                risk_penalty=request.risk_penalty,
            ),
            res.sku_id,
        )
    )
    ranked_results = ranked_results[: request.max_results]

    generated_at = datetime.now(timezone.utc).isoformat()
    response = TariffBatchScanResponseModel(
        status="OK",
        total_skus=len(request.skus),
        processed_skus=len(request.skus),
        results=ranked_results,
        skipped=skipped,
        generated_at=generated_at,
        law_context=request.law_context or DEFAULT_CONTEXT_ID,
    )
    return response
