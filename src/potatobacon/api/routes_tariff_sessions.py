from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from potatobacon.api.security import require_api_key
from potatobacon.tariff.analysis_session_store import get_default_session_store
from potatobacon.tariff.evidence_store import get_default_evidence_store
from potatobacon.tariff.sku_dossier import build_sku_dossier_v2
from potatobacon.tariff.sku_models import FactOverrideModel, TariffAnalysisSessionModel, TariffSkuDossierV2Model
from potatobacon.tariff.sku_store import get_default_sku_store

router = APIRouter(
    prefix="/api",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


class SessionCreateRequest(BaseModel):
    law_context: str | None = None

    model_config = ConfigDict(extra="forbid")


class SessionRefineRequest(BaseModel):
    fact_overrides: Optional[Dict[str, FactOverrideModel]] = None
    attached_evidence_ids: list[str] | None = None
    evidence_requested: bool | None = None
    optimize: bool | None = None

    model_config = ConfigDict(extra="forbid")


class SessionRefineResponse(BaseModel):
    session: TariffAnalysisSessionModel
    dossier: TariffSkuDossierV2Model

    model_config = ConfigDict(extra="forbid")


def _require_sku(sku_id: str) -> None:
    sku_store = get_default_sku_store()
    record = sku_store.get(sku_id)
    if not record:
        raise HTTPException(status_code=404, detail="SKU not found")


def _validate_evidence_ids(evidence_ids: list[str]) -> None:
    if not evidence_ids:
        return
    store = get_default_evidence_store()
    for evidence_id in evidence_ids:
        if not store.exists(evidence_id):
            raise HTTPException(status_code=404, detail=f"Evidence not found: {evidence_id}")


@router.post("/tariff/skus/{sku_id}/sessions", response_model=TariffAnalysisSessionModel)
def create_session_endpoint(sku_id: str, request: SessionCreateRequest) -> TariffAnalysisSessionModel:
    _require_sku(sku_id)
    store = get_default_session_store()
    return store.create_session(sku_id=sku_id, law_context=request.law_context)


@router.get("/tariff/sessions/{session_id}", response_model=TariffAnalysisSessionModel)
def get_session_endpoint(session_id: str) -> TariffAnalysisSessionModel:
    store = get_default_session_store()
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/tariff/sessions/{session_id}/refine", response_model=SessionRefineResponse)
def refine_session(session_id: str, request: SessionRefineRequest) -> SessionRefineResponse:
    session_store = get_default_session_store()
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    _require_sku(session.sku_id)

    incoming_overrides: Dict[str, FactOverrideModel] = {}
    if request.fact_overrides:
        for key, value in request.fact_overrides.items():
            incoming_overrides[key] = value if isinstance(value, FactOverrideModel) else FactOverrideModel(**value)

    attached_ids = set(session.attached_evidence_ids)
    for evidence_id in request.attached_evidence_ids or []:
        attached_ids.add(evidence_id)

    override_evidence_ids = [eid for override in incoming_overrides.values() for eid in override.evidence_ids]
    _validate_evidence_ids(sorted(attached_ids.union(override_evidence_ids)))

    updated_session = session_store.update_session(
        session_id=session_id,
        fact_overrides=incoming_overrides,
        attached_evidence_ids=sorted(attached_ids),
        status="READY_TO_OPTIMIZE",
    )

    dossier = build_sku_dossier_v2(
        sku_id=session.sku_id,
        law_context=updated_session.law_context,
        evidence_requested=bool(request.evidence_requested) if request.evidence_requested is not None else False,
        optimize=True if request.optimize is None else bool(request.optimize),
        fact_overrides=updated_session.fact_overrides,
        attached_evidence_ids=updated_session.attached_evidence_ids,
        session_id=updated_session.session_id,
    )
    return SessionRefineResponse(session=updated_session, dossier=dossier)
