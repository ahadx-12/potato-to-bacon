from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from potatobacon.api.security import require_api_key
from potatobacon.tariff.evidence_store import ALLOWED_CONTENT_TYPES, EvidenceRecord, get_default_evidence_store

router = APIRouter(
    prefix="/api",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/tariff/evidence/upload", response_model=EvidenceRecord)
async def upload_evidence(file: UploadFile = File(...), evidence_kind: str | None = None) -> EvidenceRecord:
    store = get_default_evidence_store()
    if not file.content_type or file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported content type")

    try:
        payload = await file.read()
        record = store.save(
            payload,
            filename=file.filename or "evidence",
            content_type=file.content_type,
            evidence_kind=evidence_kind,
        )
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    return record


@router.get("/tariff/evidence/{evidence_id}", response_model=EvidenceRecord)
def fetch_evidence_metadata(evidence_id: str) -> EvidenceRecord:
    store = get_default_evidence_store()
    record = store.get(evidence_id)
    if not record:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return record
