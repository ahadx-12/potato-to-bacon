from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.context_helpers import unknown_law_context_error
from potatobacon.api.security import require_api_key
from potatobacon.tariff.models import (
    TariffSuggestRequestModel,
    TariffSuggestResponseModel,
)
from potatobacon.tariff.suggest import suggest_tariff_optimizations

router = APIRouter(
    prefix="/api/tariff",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/suggest", response_model=TariffSuggestResponseModel)
def suggest_tariff_endpoint(
    request: TariffSuggestRequestModel,
) -> TariffSuggestResponseModel:
    try:
        return suggest_tariff_optimizations(request)
    except KeyError as exc:
        attempted = exc.args[0] if exc.args else request.law_context
        raise unknown_law_context_error(attempted) from exc
