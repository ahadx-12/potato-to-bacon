"""Portfolio dashboard API endpoints.

Provides the data layer for the importer dashboard:
  GET  /v1/portfolio/summary     → Total duty exposure, optimization opportunities
  GET  /v1/portfolio/skus        → All SKUs with current classifications + duty rates
  GET  /v1/portfolio/alerts      → Schedule change alerts for affected SKUs
  POST /v1/portfolio/scan        → Re-scan entire portfolio against current HTS data
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, get_registry, resolve_tenant_from_request
from potatobacon.tariff.sku_store import get_default_sku_store

# Sprint E: PostgreSQL integration
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"

if USE_POSTGRES:
    from potatobacon.db.models import Alert as AlertModel
    from potatobacon.db.session import get_standalone_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class PortfolioSKUItem(BaseModel):
    """Single SKU in the portfolio view."""

    sku_id: str
    description: Optional[str] = None
    current_hts: Optional[str] = None
    inferred_category: Optional[str] = None
    origin_country: Optional[str] = None
    declared_value_per_unit: Optional[float] = None
    annual_volume: Optional[int] = None
    current_duty_rate: Optional[float] = None
    estimated_annual_duty: Optional[float] = None
    optimization_status: Optional[str] = None
    last_analyzed: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class PortfolioSummary(BaseModel):
    """Aggregate portfolio metrics."""

    tenant_id: str
    total_skus: int
    analyzed_skus: int
    total_annual_volume: int
    total_declared_value: float
    estimated_annual_duty: float
    optimization_opportunities: int
    potential_annual_savings: float
    pending_alerts: int
    generated_at: str

    model_config = ConfigDict(extra="forbid")


class ScheduleAlert(BaseModel):
    """Alert for an HTS schedule change affecting a tenant's SKU."""

    alert_id: str
    sku_id: str
    current_hts: str
    change_type: str  # "rate_changed" | "added" | "removed" | "description_changed"
    old_rate: Optional[str] = None
    new_rate: Optional[str] = None
    description: Optional[str] = None
    detected_at: str
    acknowledged: bool = False

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# In-memory alert store (MVP)
# ---------------------------------------------------------------------------
_alerts: Dict[str, List[ScheduleAlert]] = {}  # tenant_id -> alerts


def register_alert(tenant_id: str, alert: ScheduleAlert) -> None:
    """Register a schedule change alert (PostgreSQL or in-memory)."""
    if USE_POSTGRES:
        with get_standalone_session() as session:
            alert_model = AlertModel(
                alert_id=alert.alert_id,
                tenant_id=tenant_id,
                sku_id=alert.sku_id,
                change_type=alert.change_type,
                current_hts=alert.current_hts,
                old_rate=alert.old_rate,
                new_rate=alert.new_rate,
                description=alert.description,
                detected_at=datetime.fromisoformat(alert.detected_at) if isinstance(alert.detected_at, str) else alert.detected_at,
                acknowledged=alert.acknowledged,
            )
            session.add(alert_model)
            session.commit()
    else:
        if tenant_id not in _alerts:
            _alerts[tenant_id] = []
        _alerts[tenant_id].append(alert)


def get_alerts(tenant_id: str, acknowledged: Optional[bool] = None) -> List[ScheduleAlert]:
    """Get all alerts for a tenant (PostgreSQL or in-memory)."""
    if USE_POSTGRES:
        with get_standalone_session() as session:
            query = session.query(AlertModel).filter_by(tenant_id=tenant_id)

            if acknowledged is not None:
                query = query.filter_by(acknowledged=acknowledged)

            alerts = query.order_by(AlertModel.detected_at.desc()).limit(100).all()

            return [
                ScheduleAlert(
                    alert_id=a.alert_id,
                    sku_id=a.sku_id,
                    current_hts=a.current_hts,
                    change_type=a.change_type,
                    old_rate=a.old_rate,
                    new_rate=a.new_rate,
                    description=a.description,
                    detected_at=a.detected_at.isoformat(),
                    acknowledged=a.acknowledged,
                )
                for a in alerts
            ]
    else:
        alerts = _alerts.get(tenant_id, [])
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        return alerts


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/summary", response_model=PortfolioSummary)
def portfolio_summary(
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> PortfolioSummary:
    """Aggregate portfolio metrics for the dashboard."""
    store = get_default_sku_store()
    all_skus = store.list(limit=200)

    total_volume = 0
    total_value = 0.0
    estimated_duty = 0.0
    analyzed = 0
    opportunities = 0

    for sku in all_skus:
        vol = sku.annual_volume or 0
        val = sku.declared_value_per_unit or 0.0
        total_volume += vol
        total_value += val * vol

        meta = sku.metadata or {}
        if meta.get("last_analyzed"):
            analyzed += 1
        if meta.get("optimization_available"):
            opportunities += 1

        duty_rate = meta.get("current_duty_rate", 0.0) or 0.0
        estimated_duty += val * vol * duty_rate / 100.0

    alerts = get_alerts(tenant.tenant_id)

    return PortfolioSummary(
        tenant_id=tenant.tenant_id,
        total_skus=len(all_skus),
        analyzed_skus=analyzed,
        total_annual_volume=total_volume,
        total_declared_value=round(total_value, 2),
        estimated_annual_duty=round(estimated_duty, 2),
        optimization_opportunities=opportunities,
        potential_annual_savings=0.0,  # Computed during scan
        pending_alerts=len([a for a in alerts if not a.acknowledged]),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/skus", response_model=List[PortfolioSKUItem])
def portfolio_skus(
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
    limit: int = Query(default=50, ge=1, le=200),
    prefix: Optional[str] = Query(default=None),
) -> List[PortfolioSKUItem]:
    """List all SKUs with classification and duty info for the dashboard."""
    store = get_default_sku_store()
    skus = store.list(prefix=prefix, limit=limit)

    items: List[PortfolioSKUItem] = []
    for sku in skus:
        meta = sku.metadata or {}
        vol = sku.annual_volume or 0
        val = sku.declared_value_per_unit or 0.0
        duty_rate = meta.get("current_duty_rate")
        estimated_annual = round(val * vol * (duty_rate or 0) / 100.0, 2) if duty_rate else None

        items.append(PortfolioSKUItem(
            sku_id=sku.sku_id,
            description=sku.description,
            current_hts=sku.current_hts,
            inferred_category=sku.inferred_category,
            origin_country=sku.origin_country,
            declared_value_per_unit=sku.declared_value_per_unit,
            annual_volume=sku.annual_volume,
            current_duty_rate=duty_rate,
            estimated_annual_duty=estimated_annual,
            optimization_status=meta.get("optimization_status"),
            last_analyzed=meta.get("last_analyzed"),
        ))

    return items


@router.get("/alerts", response_model=List[ScheduleAlert])
def portfolio_alerts(
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
    acknowledged: Optional[bool] = Query(default=None),
) -> List[ScheduleAlert]:
    """Get schedule change alerts affecting this tenant's SKUs."""
    return get_alerts(tenant.tenant_id, acknowledged=acknowledged)


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: str,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> Dict[str, Any]:
    """Acknowledge a schedule change alert."""
    if USE_POSTGRES:
        with get_standalone_session() as session:
            alert = session.query(AlertModel).filter_by(
                alert_id=alert_id,
                tenant_id=tenant.tenant_id
            ).first()

            if not alert:
                raise HTTPException(status_code=404, detail="Alert not found")

            alert.acknowledged = True
            alert.acknowledged_at = datetime.now(timezone.utc)
            session.commit()

            return {"status": "OK", "alert_id": alert_id}
    else:
        alerts = get_alerts(tenant.tenant_id)
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return {"status": "OK", "alert_id": alert_id}
        raise HTTPException(status_code=404, detail="Alert not found")
