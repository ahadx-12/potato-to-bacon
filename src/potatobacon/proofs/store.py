from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from potatobacon.proofs.canonical import canonical_json

# Sprint E: PostgreSQL + S3 integration
USE_POSTGRES = os.getenv("PTB_STORAGE_BACKEND", "jsonl").lower() == "postgres"
USE_S3 = os.getenv("PTB_EVIDENCE_BACKEND", "local").lower() == "s3"

if USE_POSTGRES:
    from potatobacon.db.models import Proof as ProofModel
    from potatobacon.db.session import get_standalone_session

    if USE_S3:
        from potatobacon.storage.s3_backend import get_s3_backend

def _default_path() -> Path:
    base = Path(os.getenv("PTB_DATA_ROOT", "."))
    return base / "data" / "proofs.jsonl"


class ProofStore:
    """JSONL-backed proof persistence with simple append + scan semantics."""

    def __init__(self, path: Path | None = None):
        self.path = path or _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_proof(self, proof: Dict[str, Any]) -> str:
        proof_id = proof.get("proof_id")
        if not proof_id:
            raise ValueError("proof_id is required for persistence")
        serialized = canonical_json(proof)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
        return proof_id

    def get_proof(self, proof_id: str) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("proof_id") == proof_id:
                        return record
        return None


_DEFAULT_STORE: ProofStore | None = None


# ---------------------------------------------------------------------------
# PostgreSQL + S3 proof store (Sprint E)
# ---------------------------------------------------------------------------
if USE_POSTGRES:
    class PostgresProofStore:
        """PostgreSQL metadata + S3 JSON storage."""

        def __init__(self, tenant_id: str = "default"):
            self.tenant_id = tenant_id
            self.s3 = get_s3_backend() if USE_S3 else None

        def save_proof(self, proof: Dict[str, Any]) -> str:
            """Store proof in S3 and metadata in PostgreSQL."""
            proof_id = proof.get("proof_id")
            if not proof_id:
                raise ValueError("proof_id is required for persistence")

            # Compute proof ID from canonical JSON if not provided
            canonical = canonical_json(proof)
            if not proof_id:
                proof_id = hashlib.sha256(canonical.encode()).hexdigest()
                proof["proof_id"] = proof_id

            # Upload to S3 if enabled
            if self.s3:
                s3_key = self.s3.save_proof(self.tenant_id, proof_id, proof)
                s3_bucket = self.s3.bucket
            else:
                # Fallback to local storage (for testing)
                local_dir = Path(os.getenv("PTB_DATA_ROOT", ".")) / "data" / "proofs"
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / f"{proof_id}.json"
                local_path.write_text(canonical, encoding="utf-8")
                s3_key = f"local/{self.tenant_id}/proofs/{proof_id}.json"
                s3_bucket = "local"

            # Store metadata in PostgreSQL
            with get_standalone_session() as session:
                # Check if already exists
                existing = session.query(ProofModel).filter_by(proof_id=proof_id).first()
                if existing:
                    return proof_id

                metadata = ProofModel(
                    proof_id=proof_id,
                    tenant_id=self.tenant_id,
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                    sku_id=proof.get("sku_id"),
                    baseline_hts=proof.get("baseline_hts"),
                    optimized_hts=proof.get("optimized_hts"),
                    duty_savings=proof.get("duty_savings"),
                    summary_json=proof.get("summary"),
                )
                session.add(metadata)
                session.commit()

            return proof_id

        def get_proof(self, proof_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve proof from S3 or local storage."""
            with get_standalone_session() as session:
                metadata = session.query(ProofModel).filter_by(proof_id=proof_id).first()
                if not metadata:
                    return None

                # Load from S3 if enabled
                if self.s3 and metadata.s3_bucket != "local":
                    return self.s3.get_proof(self.tenant_id, proof_id)
                else:
                    # Load from local filesystem
                    local_path = Path(os.getenv("PTB_DATA_ROOT", ".")) / "data" / "proofs" / f"{proof_id}.json"
                    if local_path.exists():
                        return json.loads(local_path.read_text(encoding="utf-8"))
                    return None

        def list_proofs(self, sku_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
            """List proof metadata from PostgreSQL."""
            with get_standalone_session() as session:
                query = session.query(ProofModel).filter_by(tenant_id=self.tenant_id)

                if sku_id:
                    query = query.filter_by(sku_id=sku_id)

                proofs = query.order_by(ProofModel.created_at.desc()).limit(limit).all()

                return [
                    {
                        "proof_id": p.proof_id,
                        "sku_id": p.sku_id,
                        "baseline_hts": p.baseline_hts,
                        "optimized_hts": p.optimized_hts,
                        "duty_savings": float(p.duty_savings) if p.duty_savings else None,
                        "created_at": p.created_at.isoformat(),
                        "summary": p.summary_json,
                    }
                    for p in proofs
                ]


def get_default_store(tenant_id: str = "default") -> ProofStore:
    """Return a module-level proof store instance (PostgreSQL+S3 or JSONL).

    Sprint E: Returns PostgresProofStore when PTB_STORAGE_BACKEND=postgres, otherwise JSONL.
    """
    if USE_POSTGRES:
        return PostgresProofStore(tenant_id=tenant_id)

    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = ProofStore()
    return _DEFAULT_STORE
