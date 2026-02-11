"""Storage backends for TEaaS production infrastructure.

Provides S3-based storage for proofs, evidence, and archives.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from potatobacon.storage.s3_backend import S3Backend, get_s3_backend

_MANIFEST_DIR = Path(os.getenv("CALE_MANIFEST_DIR", "data/manifests"))


def save_manifest(domain: str, manifest: Dict[str, Any]) -> str:
    """Persist a manifest dict to disk and return its SHA-256 hash."""
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(manifest, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = _MANIFEST_DIR / f"{domain}_{digest[:16]}.json"
    path.write_text(json.dumps(manifest, indent=2))
    latest = _MANIFEST_DIR / f"{domain}_latest.json"
    latest.write_text(json.dumps({"hash": digest, "path": str(path)}))
    return digest


def load_manifest(domain: str, manifest_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a manifest by domain (and optionally hash). Returns None if not found."""
    if manifest_hash:
        for p in _MANIFEST_DIR.glob(f"{domain}_{manifest_hash[:16]}*.json"):
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    latest = _MANIFEST_DIR / f"{domain}_latest.json"
    if latest.exists():
        try:
            meta = json.loads(latest.read_text())
            target = Path(meta.get("path", ""))
            if target.exists():
                return json.loads(target.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


def latest_manifest_hash(domain: Optional[str] = None) -> Optional[str]:
    """Return the hash of the most recent manifest for *domain*, or None."""
    if not domain:
        return None
    latest = _MANIFEST_DIR / f"{domain}_latest.json"
    if latest.exists():
        try:
            meta = json.loads(latest.read_text())
            return meta.get("hash")
        except (json.JSONDecodeError, OSError):
            pass
    return None


__all__ = [
    "S3Backend",
    "get_s3_backend",
    "save_manifest",
    "load_manifest",
    "latest_manifest_hash",
]
