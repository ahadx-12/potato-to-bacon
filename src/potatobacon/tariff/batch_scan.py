from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from potatobacon.tariff.models import (
    TariffBatchScanRequestModel,
    TariffBatchScanResponseModel,
    TariffBatchSkuResultModel,
    TariffSuggestRequestModel,
    TariffSuggestionItemModel,
)
from potatobacon.tariff.suggest import suggest_tariff_optimizations


def _score_suggestion(best: TariffSuggestionItemModel | None) -> float:
    if best is None:
        return 0.0
    if best.annual_savings_value is not None:
        return best.annual_savings_value
    if best.savings_per_unit_value is not None:
        return best.savings_per_unit_value
    return 0.0


def batch_scan_tariffs(request: TariffBatchScanRequestModel) -> TariffBatchScanResponseModel:
    """Run tariff suggestion engine across multiple SKUs and rank the outcomes."""

    ranked_results: List[TariffBatchSkuResultModel] = []
    skipped: List[TariffBatchSkuResultModel] = []

    for sku in request.skus:
        try:
            context = sku.law_context or request.law_context
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

            if suggest_response.status == "NO_CANDIDATES" or not suggest_response.suggestions:
                suggestions = suggest_response.suggestions if request.include_all_suggestions else None
                skipped.append(
                    TariffBatchSkuResultModel(
                        sku_id=sku.sku_id,
                        description=sku.description,
                        status="NO_CANDIDATES",
                        law_context=suggest_response.law_context,
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
            rank_score = _score_suggestion(best)

            ranked_results.append(
                TariffBatchSkuResultModel(
                    sku_id=sku.sku_id,
                    description=sku.description,
                    status="OK",
                    law_context=suggest_response.law_context,
                    baseline_scenario=suggest_response.baseline_scenario,
                    best=best,
                    suggestions=suggestions if request.include_all_suggestions else None,
                    rank_score=rank_score,
                    error=None,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            skipped.append(
                TariffBatchSkuResultModel(
                    sku_id=sku.sku_id,
                    description=sku.description,
                    status="ERROR",
                    law_context=sku.law_context or request.law_context,
                    baseline_scenario={},
                    best=None,
                    suggestions=None,
                    rank_score=None,
                    error=str(exc),
                )
            )

    ranked_results.sort(key=lambda res: (-_score_suggestion(res.best), res.sku_id))
    ranked_results = ranked_results[: request.max_results]

    generated_at = datetime.now(timezone.utc).isoformat()
    response = TariffBatchScanResponseModel(
        status="OK",
        total_skus=len(request.skus),
        processed_skus=len(request.skus),
        results=ranked_results,
        skipped=skipped,
        generated_at=generated_at,
        law_context=request.law_context,
    )
    return response
