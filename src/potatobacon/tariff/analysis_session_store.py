from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
import os
from typing import Dict, Iterable, Optional
from uuid import uuid4

from potatobacon.proofs.canonical import canonical_json
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID
from potatobacon.tariff.sku_models import FactOverrideModel, TariffAnalysisSessionModel


def _default_path() -> Path:
    base = Path(os.getenv("PTB_DATA_ROOT", "."))
    return base / "data" / "analysis_sessions.jsonl"


class AnalysisSessionStore:
    """Thread-safe JSONL-backed store for tariff analysis sessions."""

    def __init__(self, path: Path | None = None):
        self.path = path or _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: Dict[str, TariffAnalysisSessionModel] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        raw = json.loads(line)
                        record = TariffAnalysisSessionModel(**raw)
                    except Exception:
                        continue
                    self._records[record.session_id] = record

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self._iter_sorted():
                handle.write(canonical_json(record.serializable_dict()) + "\n")

    def _iter_sorted(self) -> Iterable[TariffAnalysisSessionModel]:
        return sorted(self._records.values(), key=lambda rec: rec.session_id)

    def create_session(self, sku_id: str, law_context: str | None = None) -> TariffAnalysisSessionModel:
        now = datetime.now(timezone.utc).isoformat()
        session = TariffAnalysisSessionModel(
            session_id=uuid4().hex,
            sku_id=sku_id,
            law_context=law_context or DEFAULT_CONTEXT_ID,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._records[session.session_id] = session
            self._persist()
        return session

    def get(self, session_id: str) -> Optional[TariffAnalysisSessionModel]:
        with self._lock:
            return self._records.get(session_id)

    def update_session(
        self,
        session_id: str,
        *,
        fact_overrides: Dict[str, FactOverrideModel] | None = None,
        attached_evidence_ids: list[str] | None = None,
        status: str | None = None,
    ) -> TariffAnalysisSessionModel:
        with self._lock:
            session = self._records.get(session_id)
            if not session:
                raise KeyError(session_id)

            normalized_overrides: Dict[str, FactOverrideModel] = dict(session.fact_overrides)
            if fact_overrides:
                for key, value in fact_overrides.items():
                    normalized_overrides[key] = value if isinstance(value, FactOverrideModel) else FactOverrideModel(**value)

            evidence_ids = set(session.attached_evidence_ids)
            for evidence_id in attached_evidence_ids or []:
                evidence_ids.add(evidence_id)

            updated_fields = {
                "fact_overrides": normalized_overrides,
                "attached_evidence_ids": sorted(evidence_ids),
                "status": status or session.status,
            }
            serialized_before = canonical_json(session.serializable_dict())
            candidate = TariffAnalysisSessionModel(
                **session.model_dump(exclude={"fact_overrides", "attached_evidence_ids", "status", "updated_at"}),
                **updated_fields,
                updated_at=session.updated_at,
            )
            serialized_after = canonical_json(candidate.serializable_dict())
            if serialized_before == serialized_after:
                return session

            candidate.updated_at = datetime.now(timezone.utc).isoformat()
            self._records[session_id] = candidate
            self._persist()
            return candidate

    def list(self, sku_id: str | None = None) -> list[TariffAnalysisSessionModel]:
        with self._lock:
            sessions = list(self._iter_sorted())
            if sku_id:
                sessions = [sess for sess in sessions if sess.sku_id == sku_id]
            return sessions


_DEFAULT_SESSION_STORE: Optional[AnalysisSessionStore] = None


def get_default_session_store(path: Path | None = None) -> AnalysisSessionStore:
    """Return a singleton analysis session store."""

    global _DEFAULT_SESSION_STORE
    resolved_path = path or _default_path()
    if _DEFAULT_SESSION_STORE is None or _DEFAULT_SESSION_STORE.path != resolved_path:
        _DEFAULT_SESSION_STORE = AnalysisSessionStore(resolved_path)
    return _DEFAULT_SESSION_STORE
