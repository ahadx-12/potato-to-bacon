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


def get_default_sku_store(path: Path | None = None) -> SKUStore:
    """Return a singleton SKU store."""

    global _DEFAULT_STORE
    target = path or _default_path()
    if _DEFAULT_STORE is None or _DEFAULT_STORE.path != target:
        _DEFAULT_STORE = SKUStore(target)
    return _DEFAULT_STORE


def get_tenant_sku_store(tenant_id: str) -> SKUStore:
    """Return a tenant-scoped SKU store.

    Each tenant gets its own JSONL file under ``data/tenants/{tenant_id}/skus.jsonl``.
    Falls back to the default store for the 'default' tenant.
    """
    if tenant_id == "default":
        return get_default_sku_store()

    if tenant_id not in _TENANT_STORES:
        base = Path(os.getenv("PTB_DATA_ROOT", "."))
        tenant_path = base / "data" / "tenants" / tenant_id / "skus.jsonl"
        _TENANT_STORES[tenant_id] = SKUStore(tenant_path)
    return _TENANT_STORES[tenant_id]
