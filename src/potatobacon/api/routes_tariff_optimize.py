from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.context_helpers import unknown_law_context_error
from potatobacon.api.security import require_api_key
from potatobacon.tariff.models import (
    TariffOptimizationRequestModel,
    TariffOptimizationResponseModel,
)
from potatobacon.tariff.models import TariffFeasibility
from potatobacon.tariff.optimizer import compute_net_savings_projection, optimize_tariff


router = APIRouter(
    prefix="/api/tariff",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/optimize", response_model=TariffOptimizationResponseModel)
def optimize_tariff_endpoint(
    request: TariffOptimizationRequestModel,
) -> TariffOptimizationResponseModel:
    try:
        result = optimize_tariff(
            base_facts=request.scenario,
            candidate_mutations=request.candidate_mutations,
            law_context=request.law_context,
            seed=request.seed or 2025,
        )
    except KeyError as exc:
        attempted = exc.args[0] if exc.args else request.law_context
        raise unknown_law_context_error(attempted) from exc

    rate_delta = result.baseline_rate - result.optimized_rate
    declared_value = request.declared_value_per_unit or 100.0
    savings_per_unit_value = rate_delta / 100.0 * declared_value

    annual_savings_value = None
    if request.annual_volume is not None:
        annual_savings_value = savings_per_unit_value * request.annual_volume
    feasibility = TariffFeasibility()
    net_savings = compute_net_savings_projection(
        baseline_rate=result.baseline_rate,
        optimized_rate=result.optimized_rate,
        declared_value_per_unit=declared_value,
        annual_volume=request.annual_volume,
        feasibility=feasibility,
    )
    ranking_score = (
        net_savings.net_annual_savings
        if net_savings.net_annual_savings is not None
        else (annual_savings_value if annual_savings_value is not None else savings_per_unit_value)
    )

    return TariffOptimizationResponseModel(
        status=result.status,
        baseline_duty_rate=result.baseline_rate,
        optimized_duty_rate=result.optimized_rate,
        savings_per_unit=rate_delta,
        best_mutation=result.best_mutation,
        baseline_scenario=result.baseline_scenario.facts,
        optimized_scenario=result.optimized_scenario.facts,
        active_codes_baseline=result.active_codes_baseline,
        active_codes_optimized=result.active_codes_optimized,
        law_context=result.law_context,
        tariff_manifest_hash=result.tariff_manifest_hash,
        proof_id=result.proof_id,
        proof_payload_hash=result.proof_payload_hash,
        provenance_chain=result.provenance_chain,
        declared_value_per_unit=declared_value,
        savings_per_unit_rate=rate_delta,
        savings_per_unit_value=savings_per_unit_value,
        annual_volume=request.annual_volume,
        annual_savings_value=annual_savings_value,
        feasibility=feasibility,
        net_savings=net_savings,
        ranking_score=ranking_score,
    )
