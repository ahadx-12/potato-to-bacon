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

# Sprint E: S3 integration
USE_S3 = os.getenv("PTB_EVIDENCE_BACKEND", "local").lower() == "s3"

if USE_S3:
    from potatobacon.storage.s3_backend import get_s3_backend
    from potatobacon.db.models import EvidenceMetadata
    from potatobacon.db.session import get_standalone_session


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/json",
    "text/plain",
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
    evidence_kind: str | None = Field(
        default=None,
        description="Optional hint about the evidence type (bom_csv, spec_sheet, cert)",
    )

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

    def save(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str,
        evidence_kind: str | None = None,
    ) -> EvidenceRecord:
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
            evidence_kind=evidence_kind,
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


# ---------------------------------------------------------------------------
# S3-backed evidence store (Sprint E)
# ---------------------------------------------------------------------------
if USE_S3:
    class S3EvidenceStore:
        """S3-backed evidence store with PostgreSQL metadata."""

        def __init__(self, tenant_id: str = "default"):
            self.tenant_id = tenant_id
            self.s3 = get_s3_backend()

        def _validate_type(self, content_type: str) -> None:
            normalized = content_type.lower()
            if normalized not in ALLOWED_CONTENT_TYPES:
                raise ValueError(f"Unsupported content type: {content_type}")

        def save(
            self,
            content: bytes,
            *,
            filename: str,
            content_type: str,
            evidence_kind: str | None = None,
        ) -> EvidenceRecord:
            """Upload evidence to S3 and store metadata in PostgreSQL."""
            self._validate_type(content_type)

            # Compute SHA-256 for content addressing
            digest = sha256(content).hexdigest()
            now = datetime.now(timezone.utc)
            byte_length = len(content)

            # Upload to S3
            s3_key = self.s3.save_evidence(
                tenant_id=self.tenant_id,
                evidence_id=digest,
                content=content,
                content_type=content_type,
                original_filename=filename,
            )

            # Store metadata in PostgreSQL
            with get_standalone_session() as session:
                # Check if already exists
                existing = session.query(EvidenceMetadata).filter_by(
                    evidence_id=digest
                ).first()

                if existing:
                    # Return existing record
                    return EvidenceRecord(
                        evidence_id=existing.evidence_id,
                        original_filename=existing.original_filename,
                        content_type=existing.content_type,
                        byte_length=existing.byte_length,
                        sha256=existing.evidence_id,
                        uploaded_at=existing.uploaded_at.isoformat(),
                        evidence_kind=existing.evidence_kind,
                    )

                # Create new metadata record
                metadata = EvidenceMetadata(
                    evidence_id=digest,
                    tenant_id=self.tenant_id,
                    original_filename=filename,
                    content_type=content_type,
                    byte_length=byte_length,
                    s3_bucket=self.s3.bucket,
                    s3_key=s3_key,
                    evidence_kind=evidence_kind,
                    uploaded_at=now,
                )
                session.add(metadata)
                session.commit()

                return EvidenceRecord(
                    evidence_id=digest,
                    original_filename=filename,
                    content_type=content_type,
                    byte_length=byte_length,
                    sha256=digest,
                    uploaded_at=now.isoformat(),
                    evidence_kind=evidence_kind,
                )

        def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
            """Get evidence metadata from PostgreSQL."""
            with get_standalone_session() as session:
                metadata = session.query(EvidenceMetadata).filter_by(
                    evidence_id=evidence_id
                ).first()

                if not metadata:
                    return None

                return EvidenceRecord(
                    evidence_id=metadata.evidence_id,
                    original_filename=metadata.original_filename,
                    content_type=metadata.content_type,
                    byte_length=metadata.byte_length,
                    sha256=metadata.evidence_id,
                    uploaded_at=metadata.uploaded_at.isoformat(),
                    evidence_kind=metadata.evidence_kind,
                )

        def exists(self, evidence_id: str) -> bool:
            """Check if evidence exists."""
            return self.get(evidence_id) is not None

        def retrieve_blob(self, evidence_id: str) -> Optional[bytes]:
            """Download evidence from S3."""
            return self.s3.get_evidence(self.tenant_id, evidence_id)

        def get_presigned_url(self, evidence_id: str, expiration: int = 3600) -> Optional[str]:
            """Generate presigned URL for direct download."""
            with get_standalone_session() as session:
                metadata = session.query(EvidenceMetadata).filter_by(
                    evidence_id=evidence_id
                ).first()

                if not metadata:
                    return None

                return self.s3.generate_presigned_url(metadata.s3_key, expiration)


def get_default_evidence_store(index_path: Path | None = None, data_dir: Path | None = None, tenant_id: str = "default") -> EvidenceStore:
    """Return a singleton evidence store (S3 or local filesystem).

    Sprint E: Returns S3EvidenceStore when PTB_EVIDENCE_BACKEND=s3, otherwise local filesystem.
    """
    if USE_S3:
        return S3EvidenceStore(tenant_id=tenant_id)

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
