"""Tenant isolation for TEaaS multi-tenant operation.

Each API key maps to a tenant.  Tenant-scoped data (SKUs, proofs,
analysis sessions) is isolated by prefixing storage keys with the
tenant ID.  For the MVP this uses a simple in-memory registry that
can be swapped for a database in production.

Sprint E: Added PostgreSQL backend support via PTB_STORAGE_BACKEND env var.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

# Sprint E: PostgreSQL integration
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"

if USE_POSTGRES:
    from potatobacon.db.models import Tenant as TenantModel, APIKey as APIKeyModel
    from potatobacon.db.session import get_standalone_session


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Tenant:
        return cls(**data)


# ---------------------------------------------------------------------------
# Registry (in-memory for MVP, PostgreSQL for production)
# ---------------------------------------------------------------------------
class TenantRegistry:
    """Maps API keys to tenants and enforces rate limits."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._tenants: Dict[str, Tenant] = {}
        self._key_to_tenant: Dict[str, str] = {}
        self.path = path or Path(os.getenv("PTB_DATA_ROOT", ".")) / "data" / "tenants.json"
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for t_data in data.get("tenants", []):
                    tenant = Tenant.from_dict(t_data)
                    self._tenants[tenant.tenant_id] = tenant
                    for key in tenant.api_keys:
                        self._key_to_tenant[key] = tenant.tenant_id
        except Exception:
            pass

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"tenants": [t.to_dict() for t in self._tenants.values()]}
        with self._lock:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

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
        with self._lock:
            self._tenants[tenant_id] = tenant
            self._key_to_tenant[api_key] = tenant_id
            self._persist()

        # Also register in security layer if running in same process
        try:
            from potatobacon.api.security import register_api_key
            register_api_key(api_key)
        except ImportError:
            pass

        return tenant

    def add_api_key(self, tenant_id: str, api_key: str) -> None:
        """Associate an additional API key with a tenant."""
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError(f"Unknown tenant: {tenant_id}")
            self._tenants[tenant_id].api_keys.append(api_key)
            self._key_to_tenant[api_key] = tenant_id
            self._persist()

    def resolve(self, api_key: str) -> Optional[Tenant]:
        """Look up the tenant for an API key."""
        tenant_id = self._key_to_tenant.get(api_key)
        if tenant_id is None:
            # Try reloading in case another process updated it
            self._load()
            tenant_id = self._key_to_tenant.get(api_key)

        if tenant_id is None:
            return None
        return self._tenants.get(tenant_id)

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Look up a tenant by ID."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
             # Try reloading
            self._load()
            tenant = self._tenants.get(tenant_id)
        return tenant

    def increment_usage(self, tenant_id: str) -> None:
        """Increment the monthly analysis counter."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant:
                tenant.monthly_analyses += 1
                self._persist()

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
# PostgreSQL-backed registry (Sprint E)
# ---------------------------------------------------------------------------
if USE_POSTGRES:
    class PostgresTenantRegistry:
        """PostgreSQL-backed tenant registry."""

        def register_tenant(
            self,
            tenant_id: str,
            name: str,
            api_key: str,
            plan: str = "starter",
        ) -> Tenant:
            """Create a new tenant and API key in PostgreSQL."""
            with get_standalone_session() as session:
                # Create tenant
                tenant_model = TenantModel(
                    tenant_id=tenant_id,
                    name=name,
                    plan=plan,
                    sku_limit=_plan_limits(plan)["sku_limit"],
                    monthly_analysis_limit=_plan_limits(plan)["monthly_analyses"],
                )
                session.add(tenant_model)

                # Create API key
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                api_key_model = APIKeyModel(
                    tenant_id=tenant_id,
                    key_hash=key_hash,
                    description="Primary API key",
                )
                session.add(api_key_model)
                session.commit()

                return Tenant(
                    tenant_id=tenant_id,
                    name=name,
                    api_keys=[api_key],
                    plan=plan,
                    sku_limit=tenant_model.sku_limit,
                    monthly_analyses=tenant_model.monthly_analyses,
                    monthly_analysis_limit=tenant_model.monthly_analysis_limit,
                    metadata=tenant_model.metadata_json or {},
                )

        def add_api_key(self, tenant_id: str, api_key: str) -> None:
            """Associate an additional API key with a tenant."""
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            with get_standalone_session() as session:
                tenant = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
                if not tenant:
                    raise KeyError(f"Unknown tenant: {tenant_id}")

                api_key_model = APIKeyModel(
                    tenant_id=tenant_id,
                    key_hash=key_hash,
                    description="Additional API key",
                )
                session.add(api_key_model)
                session.commit()

        def resolve(self, api_key: str) -> Optional[Tenant]:
            """Look up tenant by API key from PostgreSQL."""
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            with get_standalone_session() as session:
                api_key_record = session.query(APIKeyModel).filter_by(
                    key_hash=key_hash,
                    revoked=False
                ).first()

                if not api_key_record:
                    return None

                tenant_model = session.query(TenantModel).filter_by(
                    tenant_id=api_key_record.tenant_id
                ).first()

                if not tenant_model:
                    return None

                return Tenant(
                    tenant_id=tenant_model.tenant_id,
                    name=tenant_model.name,
                    api_keys=[],  # Not needed for resolution
                    plan=tenant_model.plan,
                    created_at=tenant_model.created_at.isoformat(),
                    sku_limit=tenant_model.sku_limit,
                    monthly_analyses=tenant_model.monthly_analyses,
                    monthly_analysis_limit=tenant_model.monthly_analysis_limit,
                    metadata=tenant_model.metadata_json or {},
                )

        def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
            """Look up a tenant by ID."""
            with get_standalone_session() as session:
                tenant_model = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
                if not tenant_model:
                    return None

                return Tenant(
                    tenant_id=tenant_model.tenant_id,
                    name=tenant_model.name,
                    api_keys=[],
                    plan=tenant_model.plan,
                    created_at=tenant_model.created_at.isoformat(),
                    sku_limit=tenant_model.sku_limit,
                    monthly_analyses=tenant_model.monthly_analyses,
                    monthly_analysis_limit=tenant_model.monthly_analysis_limit,
                    metadata=tenant_model.metadata_json or {},
                )

        def increment_usage(self, tenant_id: str) -> None:
            """Increment the monthly analysis counter."""
            with get_standalone_session() as session:
                tenant = session.query(TenantModel).filter_by(tenant_id=tenant_id).first()
                if tenant:
                    tenant.monthly_analyses += 1
                    session.commit()

        def check_rate_limit(self, tenant: Tenant) -> bool:
            """Return True if the tenant is within their monthly limit."""
            return tenant.monthly_analyses < tenant.monthly_analysis_limit


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------
_registry = TenantRegistry()


def _seed_default_tenant() -> None:
    """Create a default tenant for development / single-tenant mode."""
    default_key = os.getenv("PTB_API_KEY", "dev-key-local")
    registry = get_registry()
    if registry.resolve(default_key) is None:
        registry.register_tenant(
            tenant_id="default",
            name="Development Tenant",
            api_key=default_key,
            plan="enterprise",
        )


def get_registry() -> TenantRegistry:
    """Return the global tenant registry (PostgreSQL or in-memory)."""
    if USE_POSTGRES:
        return PostgresTenantRegistry()
    return _registry


_seed_default_tenant()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def resolve_tenant_from_request(request: Request) -> Tenant:
    """FastAPI dependency that extracts the tenant from the API key header.

    Returns the Tenant or raises 401/429.
    """
    from fastapi import HTTPException

    api_key = request.headers.get("X-API-Key") or os.getenv("PTB_API_KEY", "dev-key-local")
    registry = get_registry()
    tenant = registry.resolve(api_key)
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not registry.check_rate_limit(tenant):
        raise HTTPException(
            status_code=429,
            detail=f"Monthly analysis limit reached ({tenant.monthly_analysis_limit})",
        )
    return tenant


def tenant_storage_prefix(tenant: Tenant) -> str:
    """Return a storage path prefix for tenant-scoped data."""
    return f"tenants/{tenant.tenant_id}"
