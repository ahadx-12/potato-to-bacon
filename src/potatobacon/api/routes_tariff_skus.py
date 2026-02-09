from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, resolve_tenant_from_request
from potatobacon.tariff.sku_models import SKURecordModel
from potatobacon.tariff.sku_store import get_tenant_sku_store

router = APIRouter(
    prefix="/api",
    tags=["tariff"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/tariff/skus")
def upsert_sku(
    record: SKURecordModel,
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    store = get_tenant_sku_store(tenant.tenant_id)
    existing = store.get(record.sku_id)
    saved = store.upsert(record.sku_id, record)
    created = existing is None
    return {
        "status": "OK",
        "sku_id": saved.sku_id,
        "tenant_id": tenant.tenant_id,
        "created": created,
        "updated_at": saved.updated_at,
        "created_at": saved.created_at,
    }


@router.get("/tariff/skus/{sku_id}", response_model=SKURecordModel)
def fetch_sku(
    sku_id: str,
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    store = get_tenant_sku_store(tenant.tenant_id)
    record = store.get(sku_id)
    if not record:
        raise HTTPException(status_code=404, detail="SKU not found")
    return record


@router.get("/tariff/skus", response_model=list[SKURecordModel])
def list_skus(
    prefix: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    store = get_tenant_sku_store(tenant.tenant_id)
    return store.list(prefix=prefix, limit=limit)


@router.delete("/tariff/skus/{sku_id}")
def delete_sku(
    sku_id: str,
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    store = get_tenant_sku_store(tenant.tenant_id)
    removed = store.delete(sku_id)
    if not removed:
        raise HTTPException(status_code=404, detail="SKU not found")
    return {"status": "DELETED", "sku_id": sku_id, "tenant_id": tenant.tenant_id}
