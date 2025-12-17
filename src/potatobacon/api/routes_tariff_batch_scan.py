from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.context_helpers import unknown_law_context_error
from potatobacon.api.security import require_api_key
from potatobacon.tariff.batch_scan import batch_scan_tariffs
from potatobacon.tariff.models import (
    TariffBatchScanRequestModel,
    TariffBatchScanResponseModel,
)

router = APIRouter(
    prefix="/api/tariff",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/batch-scan", response_model=TariffBatchScanResponseModel)
def batch_scan_endpoint(
    request: TariffBatchScanRequestModel,
) -> TariffBatchScanResponseModel:
    try:
        return batch_scan_tariffs(request)
    except KeyError as exc:
        attempted = exc.args[0] if exc.args else request.law_context
        raise unknown_law_context_error(attempted) from exc
