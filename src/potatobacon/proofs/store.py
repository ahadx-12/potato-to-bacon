from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional


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
        serialized = json.dumps(proof, sort_keys=True)
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


def get_default_store() -> ProofStore:
    """Return a module-level proof store instance."""

    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = ProofStore()
    return _DEFAULT_STORE
