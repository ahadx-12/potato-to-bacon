from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.security import require_api_key
from potatobacon.tariff.engine import explain_tariff_scenario
from potatobacon.tariff.models import TariffExplainResponseModel, TariffHuntRequestModel

router = APIRouter(
    prefix="/v1/tariff",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/explain", response_model=TariffExplainResponseModel)
def explain_tariff(request: TariffHuntRequestModel) -> TariffExplainResponseModel:
    """Return SAT/UNSAT explanations for tariff scenarios."""

    return explain_tariff_scenario(
        base_facts=request.scenario,
        mutations=request.mutations,
        law_context=request.law_context,
    )
