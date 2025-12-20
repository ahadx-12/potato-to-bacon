from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from potatobacon.api.security import require_api_key
from potatobacon.tariff.sku_dossier import build_sku_dossier_v2
from potatobacon.tariff.sku_models import FactOverrideModel, TariffSkuDossierV2Model

router = APIRouter(
    prefix="/api",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


class SKUDossierRequestModel(BaseModel):
    law_context: str | None = None
    evidence_requested: bool | None = None
    optimize: bool | None = True
    risk_penalize: bool | None = None
    fact_overrides: dict[str, FactOverrideModel] | None = None
    attached_evidence_ids: list[str] | None = None
    analysis_session_id: str | None = None

    model_config = ConfigDict(extra="forbid")


@router.post("/tariff/skus/{sku_id}/dossier", response_model=TariffSkuDossierV2Model)
def build_sku_dossier_endpoint(sku_id: str, request: SKUDossierRequestModel) -> TariffSkuDossierV2Model:
    try:
        return build_sku_dossier_v2(
            sku_id=sku_id,
            law_context=request.law_context,
            evidence_requested=bool(request.evidence_requested) if request.evidence_requested is not None else False,
            optimize=True if request.optimize is None else bool(request.optimize),
            risk_penalize=request.risk_penalize,
            fact_overrides=request.fact_overrides,
            attached_evidence_ids=request.attached_evidence_ids,
            session_id=request.analysis_session_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="SKU not found")
