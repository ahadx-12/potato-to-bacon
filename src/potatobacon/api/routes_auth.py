"""Authentication and tenant identity endpoints.

Provides /v1/auth/whoami so clients can verify their API key
and inspect tenant metadata, plan limits, and usage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, resolve_tenant_from_request

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class WhoAmIResponse(BaseModel):
    """Identity and plan details for the authenticated tenant."""

    tenant_id: str
    name: str
    plan: str
    sku_limit: int
    monthly_analysis_limit: int
    monthly_analyses_used: int
    created_at: str

    model_config = ConfigDict(extra="forbid")


@router.get("/whoami", response_model=WhoAmIResponse)
def whoami(
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> WhoAmIResponse:
    """Return identity and plan metadata for the authenticated API key."""
    return WhoAmIResponse(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        plan=tenant.plan,
        sku_limit=tenant.sku_limit,
        monthly_analysis_limit=tenant.monthly_analysis_limit,
        monthly_analyses_used=tenant.monthly_analyses,
        created_at=tenant.created_at,
    )
