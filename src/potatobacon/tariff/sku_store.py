from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from potatobacon.proofs.canonical import canonical_json
from potatobacon.tariff.category_detector import CategoryDetector
from potatobacon.tariff.sku_models import SKURecordModel

# Sprint E: PostgreSQL integration
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"

if USE_POSTGRES:
    from potatobacon.db.models import SKU as SKUModel
    from potatobacon.db.session import get_standalone_session


def _default_path() -> Path:
    base = Path(os.getenv("PTB_DATA_ROOT", "."))
    return base / "data" / "skus.jsonl"


class SKUStore:
    """Thread-safe JSONL-backed SKU registry with deterministic serialization."""

    def __init__(self, path: Path | None = None):
        self.path = path or _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: Dict[str, SKURecordModel] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        raw = json.loads(line)
                        record = SKURecordModel(**raw)
                    except Exception:
                        continue
                    self._records[record.sku_id] = record

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self._iter_sorted():
                handle.write(canonical_json(record.serializable_dict()) + "\n")

    def _iter_sorted(self) -> Iterable[SKURecordModel]:
        return sorted(self._records.values(), key=lambda rec: rec.sku_id)

    def upsert(self, sku_id: str, payload: Dict | SKURecordModel) -> SKURecordModel:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._records.get(sku_id)
            incoming_payload = payload.serializable_dict() if isinstance(payload, SKURecordModel) else dict(payload)
            merged_payload = existing.serializable_dict() if existing else {}
            merged_payload.update(incoming_payload)
            description_changed = existing is None or merged_payload.get("description") != existing.description
            hts_changed = existing is None or merged_payload.get("current_hts") != existing.current_hts
            created_at = existing.created_at if existing else now
            if description_changed or hts_changed:
                detector = CategoryDetector()
                candidate = SKURecordModel(**{**merged_payload, "sku_id": sku_id, "created_at": created_at, "updated_at": now})
                result = detector.detect(candidate)
                merged_payload["inferred_category"] = result.primary.name
                merged_payload["category_confidence"] = result.confidence
            record = SKURecordModel(**{**merged_payload, "sku_id": sku_id, "created_at": created_at, "updated_at": now})
            self._records[sku_id] = record
            self._persist()
        return record

    def get(self, sku_id: str) -> Optional[SKURecordModel]:
        with self._lock:
            record = self._records.get(sku_id)
            return record

    def list(self, prefix: str | None = None, limit: int = 50) -> List[SKURecordModel]:
        limit = max(1, min(limit, 200))
        with self._lock:
            candidates = self._iter_sorted()
            results: List[SKURecordModel] = []
            for record in candidates:
                if prefix and not record.sku_id.startswith(prefix):
                    continue
                results.append(record)
                if len(results) >= limit:
                    break
        return results

    def delete(self, sku_id: str) -> bool:
        with self._lock:
            if sku_id not in self._records:
                return False
            del self._records[sku_id]
            self._persist()
        return True


_DEFAULT_STORE: Optional[SKUStore] = None
_TENANT_STORES: Dict[str, SKUStore] = {}


# ---------------------------------------------------------------------------
# PostgreSQL-backed SKU store (Sprint E)
# ---------------------------------------------------------------------------
if USE_POSTGRES:
    class PostgresSKUStore:
        """PostgreSQL-backed SKU store."""

        def __init__(self, tenant_id: str = "default"):
            self.tenant_id = tenant_id

        def upsert(self, sku_id: str, payload: Dict | SKURecordModel) -> SKURecordModel:
            """Insert or update SKU in PostgreSQL."""
            now = datetime.now(timezone.utc)
            incoming_payload = payload.serializable_dict() if isinstance(payload, SKURecordModel) else dict(payload)

            with get_standalone_session() as session:
                existing = session.query(SKUModel).filter_by(
                    tenant_id=self.tenant_id,
                    sku_id=sku_id
                ).first()

                merged_payload = {}
                if existing:
                    # Update existing
                    merged_payload = {
                        "description": existing.description,
                        "current_hts": existing.current_hts,
                        "origin_country": existing.origin_country,
                        "declared_value": float(existing.declared_value) if existing.declared_value else None,
                        "inferred_category": existing.inferred_category,
                        "category_confidence": float(existing.category_confidence) if existing.category_confidence else None,
                    }
                    merged_payload.update(incoming_payload)
                    created_at = existing.created_at
                else:
                    # Create new
                    merged_payload = incoming_payload
                    created_at = now

                # Re-detect category if description or HTS changed
                description_changed = existing is None or merged_payload.get("description") != (existing.description if existing else None)
                hts_changed = existing is None or merged_payload.get("current_hts") != (existing.current_hts if existing else None)

                if description_changed or hts_changed:
                    detector = CategoryDetector()
                    candidate = SKURecordModel(**{**merged_payload, "sku_id": sku_id, "created_at": created_at.isoformat(), "updated_at": now.isoformat()})
                    result = detector.detect(candidate)
                    merged_payload["inferred_category"] = result.primary.name
                    merged_payload["category_confidence"] = result.confidence

                # Update or create in database
                if existing:
                    for key, value in merged_payload.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = now
                    sku_model = existing
                else:
                    sku_model = SKUModel(
                        tenant_id=self.tenant_id,
                        sku_id=sku_id,
                        description=merged_payload.get("description"),
                        current_hts=merged_payload.get("current_hts"),
                        origin_country=merged_payload.get("origin_country"),
                        declared_value=merged_payload.get("declared_value"),
                        inferred_category=merged_payload.get("inferred_category"),
                        category_confidence=merged_payload.get("category_confidence"),
                        created_at=created_at,
                        updated_at=now,
                    )
                    session.add(sku_model)

                session.commit()

                # Return as SKURecordModel for compatibility
                return SKURecordModel(
                    sku_id=sku_model.sku_id,
                    description=sku_model.description,
                    current_hts=sku_model.current_hts,
                    origin_country=sku_model.origin_country,
                    declared_value=float(sku_model.declared_value) if sku_model.declared_value else None,
                    inferred_category=sku_model.inferred_category,
                    category_confidence=float(sku_model.category_confidence) if sku_model.category_confidence else None,
                    created_at=sku_model.created_at.isoformat(),
                    updated_at=sku_model.updated_at.isoformat(),
                )

        def get(self, sku_id: str) -> Optional[SKURecordModel]:
            """Retrieve SKU from PostgreSQL."""
            with get_standalone_session() as session:
                sku = session.query(SKUModel).filter_by(
                    tenant_id=self.tenant_id,
                    sku_id=sku_id
                ).first()

                if not sku:
                    return None

                return SKURecordModel(
                    sku_id=sku.sku_id,
                    description=sku.description,
                    current_hts=sku.current_hts,
                    origin_country=sku.origin_country,
                    declared_value=float(sku.declared_value) if sku.declared_value else None,
                    inferred_category=sku.inferred_category,
                    category_confidence=float(sku.category_confidence) if sku.category_confidence else None,
                    created_at=sku.created_at.isoformat(),
                    updated_at=sku.updated_at.isoformat(),
                )

        def list(self, prefix: str | None = None, limit: int = 50) -> List[SKURecordModel]:
            """List SKUs from PostgreSQL."""
            limit = max(1, min(limit, 200))
            with get_standalone_session() as session:
                query = session.query(SKUModel).filter_by(tenant_id=self.tenant_id)

                if prefix:
                    query = query.filter(SKUModel.sku_id.startswith(prefix))

                skus = query.order_by(SKUModel.sku_id).limit(limit).all()

                return [
                    SKURecordModel(
                        sku_id=sku.sku_id,
                        description=sku.description,
                        current_hts=sku.current_hts,
                        origin_country=sku.origin_country,
                        declared_value=float(sku.declared_value) if sku.declared_value else None,
                        inferred_category=sku.inferred_category,
                        category_confidence=float(sku.category_confidence) if sku.category_confidence else None,
                        created_at=sku.created_at.isoformat(),
                        updated_at=sku.updated_at.isoformat(),
                    )
                    for sku in skus
                ]

        def delete(self, sku_id: str) -> bool:
            """Delete SKU from PostgreSQL."""
            with get_standalone_session() as session:
                sku = session.query(SKUModel).filter_by(
                    tenant_id=self.tenant_id,
                    sku_id=sku_id
                ).first()

                if not sku:
                    return False

                session.delete(sku)
                session.commit()
                return True


def get_default_sku_store(path: Path | None = None) -> SKUStore:
    """Return a singleton SKU store (PostgreSQL or JSONL)."""
    if USE_POSTGRES:
        return PostgresSKUStore(tenant_id="default")

    global _DEFAULT_STORE
    target = path or _default_path()
    if _DEFAULT_STORE is None or _DEFAULT_STORE.path != target:
        _DEFAULT_STORE = SKUStore(target)
    return _DEFAULT_STORE


def get_tenant_sku_store(tenant_id: str) -> SKUStore:
    """Return a tenant-scoped SKU store (PostgreSQL or JSONL).

    Each tenant gets its own JSONL file under ``data/tenants/{tenant_id}/skus.jsonl``.
    Falls back to the default store for the 'default' tenant.
    """
    if USE_POSTGRES:
        return PostgresSKUStore(tenant_id=tenant_id)

    if tenant_id == "default":
        return get_default_sku_store()

    if tenant_id not in _TENANT_STORES:
        base = Path(os.getenv("PTB_DATA_ROOT", "."))
        tenant_path = base / "data" / "tenants" / tenant_id / "skus.jsonl"
        _TENANT_STORES[tenant_id] = SKUStore(tenant_path)
    return _TENANT_STORES[tenant_id]
