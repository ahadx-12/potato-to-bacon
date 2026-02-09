"""Tenant isolation for TEaaS multi-tenant operation.

Each API key maps to a tenant.  Tenant-scoped data (SKUs, proofs,
analysis sessions) is isolated by prefixing storage keys with the
tenant ID.  For the MVP this uses a simple in-memory registry that
can be swapped for a database in production.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request


# ---------------------------------------------------------------------------
# Tenant model
# ---------------------------------------------------------------------------
@dataclass
class Tenant:
    """A TEaaS customer account."""

    tenant_id: str
    name: str
    api_keys: List[str] = field(default_factory=list)
    plan: str = "starter"  # starter | professional | enterprise
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sku_limit: int = 100
    monthly_analyses: int = 0
    monthly_analysis_limit: int = 500
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry (in-memory for MVP, PostgreSQL for production)
# ---------------------------------------------------------------------------
class TenantRegistry:
    """Maps API keys to tenants and enforces rate limits."""

    def __init__(self) -> None:
        self._tenants: Dict[str, Tenant] = {}
        self._key_to_tenant: Dict[str, str] = {}

    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        api_key: str,
        plan: str = "starter",
    ) -> Tenant:
        """Create a new tenant and associate an API key."""
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            api_keys=[api_key],
            plan=plan,
            sku_limit=_plan_limits(plan)["sku_limit"],
            monthly_analysis_limit=_plan_limits(plan)["monthly_analyses"],
        )
        self._tenants[tenant_id] = tenant
        self._key_to_tenant[api_key] = tenant_id
        return tenant

    def add_api_key(self, tenant_id: str, api_key: str) -> None:
        """Associate an additional API key with a tenant."""
        if tenant_id not in self._tenants:
            raise KeyError(f"Unknown tenant: {tenant_id}")
        self._tenants[tenant_id].api_keys.append(api_key)
        self._key_to_tenant[api_key] = tenant_id

    def resolve(self, api_key: str) -> Optional[Tenant]:
        """Look up the tenant for an API key."""
        tenant_id = self._key_to_tenant.get(api_key)
        if tenant_id is None:
            return None
        return self._tenants.get(tenant_id)

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Look up a tenant by ID."""
        return self._tenants.get(tenant_id)

    def increment_usage(self, tenant_id: str) -> None:
        """Increment the monthly analysis counter."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.monthly_analyses += 1

    def check_rate_limit(self, tenant: Tenant) -> bool:
        """Return True if the tenant is within their monthly limit."""
        return tenant.monthly_analyses < tenant.monthly_analysis_limit


def _plan_limits(plan: str) -> Dict[str, int]:
    """Return limits for each pricing plan."""
    plans = {
        "starter": {"sku_limit": 100, "monthly_analyses": 500},
        "professional": {"sku_limit": 1000, "monthly_analyses": 5000},
        "enterprise": {"sku_limit": 100000, "monthly_analyses": 100000},
    }
    return plans.get(plan, plans["starter"])


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------
_registry = TenantRegistry()


def _seed_default_tenant() -> None:
    """Create a default tenant for development / single-tenant mode."""
    default_key = os.getenv("PTB_API_KEY", "dev-key-local")
    if _registry.resolve(default_key) is None:
        _registry.register_tenant(
            tenant_id="default",
            name="Development Tenant",
            api_key=default_key,
            plan="enterprise",
        )


_seed_default_tenant()


def get_registry() -> TenantRegistry:
    """Return the global tenant registry."""
    return _registry


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def resolve_tenant_from_request(request: Request) -> Tenant:
    """FastAPI dependency that extracts the tenant from the API key header.

    Returns the Tenant or raises 401/429.
    """
    api_key = request.headers.get("X-API-Key") or os.getenv("PTB_API_KEY", "dev-key-local")
    tenant = _registry.resolve(api_key)
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not _registry.check_rate_limit(tenant):
        raise HTTPException(
            status_code=429,
            detail=f"Monthly analysis limit reached ({tenant.monthly_analysis_limit})",
        )
    return tenant


def tenant_storage_prefix(tenant: Tenant) -> str:
    """Return a storage path prefix for tenant-scoped data."""
    return f"tenants/{tenant.tenant_id}"
