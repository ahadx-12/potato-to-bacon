from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from potatobacon.proofs.canonical import canonical_json


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "text/csv",
    "application/json",
}


def _default_index_path() -> Path:
    base = Path(os.getenv("PTB_DATA_ROOT", "."))
    return base / "data" / "evidence_index.jsonl"


def _default_data_dir(index_path: Path | None = None) -> Path:
    if index_path is None:
        index_path = _default_index_path()
    return index_path.parent / "evidence"


class EvidenceRecord(BaseModel):
    """Immutable evidence metadata indexed by hash-derived identifier."""

    evidence_id: str
    original_filename: str
    content_type: str
    byte_length: int
    sha256: str
    uploaded_at: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("uploaded_at", mode="before")
    @classmethod
    def _normalize_timestamp(cls, value):
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def serializable_dict(self) -> Dict[str, str | int]:
        return self.model_dump()


class EvidenceStore:
    """Hash-addressed evidence vault with deterministic JSONL index."""

    def __init__(self, index_path: Path | None = None, data_dir: Path | None = None):
        self.index_path = index_path or _default_index_path()
        self.data_dir = data_dir or _default_data_dir(self.index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: Dict[str, EvidenceRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        with self._lock:
            with self.index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = EvidenceRecord(**json.loads(line))
                    except Exception:
                        continue
                    self._records[record.evidence_id] = record

    def _persist(self) -> None:
        with self.index_path.open("w", encoding="utf-8") as handle:
            for evidence_id in sorted(self._records.keys()):
                record = self._records[evidence_id]
                handle.write(canonical_json(record.serializable_dict()) + "\n")

    def _validate_type(self, content_type: str) -> None:
        normalized = content_type.lower()
        if normalized not in ALLOWED_CONTENT_TYPES:
            raise ValueError(f"Unsupported content type: {content_type}")

    def save(self, content: bytes, *, filename: str, content_type: str) -> EvidenceRecord:
        """Persist an evidence blob and return deterministic metadata."""

        self._validate_type(content_type)
        digest = sha256(content).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        byte_length = len(content)
        record = EvidenceRecord(
            evidence_id=digest,
            original_filename=filename,
            content_type=content_type,
            byte_length=byte_length,
            sha256=digest,
            uploaded_at=now,
        )
        blob_path = self.data_dir / digest

        with self._lock:
            existing = self._records.get(digest)
            if existing:
                return existing

            if not blob_path.exists():
                blob_path.write_bytes(content)
            self._records[digest] = record
            self._persist()

        return record

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        with self._lock:
            return self._records.get(evidence_id)

    def exists(self, evidence_id: str) -> bool:
        return self.get(evidence_id) is not None


_DEFAULT_EVIDENCE_STORE: Optional[EvidenceStore] = None


def get_default_evidence_store(index_path: Path | None = None, data_dir: Path | None = None) -> EvidenceStore:
    """Return a singleton evidence store configured for the current environment."""

    global _DEFAULT_EVIDENCE_STORE
    resolved_index = index_path or _default_index_path()
    resolved_dir = data_dir or _default_data_dir(resolved_index)
    if (
        _DEFAULT_EVIDENCE_STORE is None
        or _DEFAULT_EVIDENCE_STORE.index_path != resolved_index
        or _DEFAULT_EVIDENCE_STORE.data_dir != resolved_dir
    ):
        _DEFAULT_EVIDENCE_STORE = EvidenceStore(resolved_index, resolved_dir)
    return _DEFAULT_EVIDENCE_STORE
