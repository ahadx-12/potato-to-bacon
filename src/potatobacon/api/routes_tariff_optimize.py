from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.security import require_api_key
from potatobacon.tariff.models import (
    TariffOptimizationRequestModel,
    TariffOptimizationResponseModel,
)
from potatobacon.tariff.optimizer import optimize_tariff


router = APIRouter(
    prefix="/api/tariff",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/optimize", response_model=TariffOptimizationResponseModel)
def optimize_tariff_endpoint(
    request: TariffOptimizationRequestModel,
) -> TariffOptimizationResponseModel:
    result = optimize_tariff(
        base_facts=request.scenario,
        candidate_mutations=request.candidate_mutations,
        law_context=request.law_context,
        seed=request.seed or 2025,
    )

    savings = result.baseline_rate - result.optimized_rate

    return TariffOptimizationResponseModel(
        status=result.status,
        baseline_duty_rate=result.baseline_rate,
        optimized_duty_rate=result.optimized_rate,
        savings_per_unit=savings,
        best_mutation=result.best_mutation,
        baseline_scenario=result.baseline_scenario.facts,
        optimized_scenario=result.optimized_scenario.facts,
        active_codes_baseline=result.active_codes_baseline,
        active_codes_optimized=result.active_codes_optimized,
        law_context=result.law_context,
        proof_id=result.proof_id,
        provenance_chain=result.provenance_chain,
    )
