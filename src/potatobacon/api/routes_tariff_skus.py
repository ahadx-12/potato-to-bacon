from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from potatobacon.api.security import require_api_key
from potatobacon.tariff.sku_models import SKURecordModel
from potatobacon.tariff.sku_store import get_default_sku_store

router = APIRouter(
    prefix="/api",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/tariff/skus")
def upsert_sku(record: SKURecordModel):
    store = get_default_sku_store()
    existing = store.get(record.sku_id)
    saved = store.upsert(record.sku_id, record)
    created = existing is None
    return {
        "status": "OK",
        "sku_id": saved.sku_id,
        "created": created,
        "updated_at": saved.updated_at,
        "created_at": saved.created_at,
    }


@router.get("/tariff/skus/{sku_id}", response_model=SKURecordModel)
def fetch_sku(sku_id: str):
    store = get_default_sku_store()
    record = store.get(sku_id)
    if not record:
        raise HTTPException(status_code=404, detail="SKU not found")
    return record


@router.get("/tariff/skus", response_model=list[SKURecordModel])
def list_skus(prefix: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=200)):
    store = get_default_sku_store()
    return store.list(prefix=prefix, limit=limit)


@router.delete("/tariff/skus/{sku_id}")
def delete_sku(sku_id: str):
    store = get_default_sku_store()
    removed = store.delete(sku_id)
    if not removed:
        raise HTTPException(status_code=404, detail="SKU not found")
    return {"status": "DELETED", "sku_id": sku_id}
