"""Persistent storage helpers for potato-to-bacon artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

BASE_PATH = Path(os.getenv("PTB_DATA_ROOT", "artifacts"))
SCHEMA_DIR = BASE_PATH / "schemas"
CODE_DIR = BASE_PATH / "code"
MANIFEST_DIR = BASE_PATH / "manifests"
INDEX_DIR = BASE_PATH / "index"
MANIFEST_INDEX = INDEX_DIR / "manifests.jsonl"


def _ensure_dirs() -> None:
    for directory in (SCHEMA_DIR, CODE_DIR, MANIFEST_DIR, INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2))


def _stable_digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def save_schema(schema: Dict[str, Any]) -> str:
    """Persist a schema document and return its content digest."""

    _ensure_dirs()
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    digest = _stable_digest(payload)
    (SCHEMA_DIR / f"{digest}.json").write_bytes(
        json.dumps(schema, sort_keys=True, indent=2).encode()
    )
    return digest


def load_schema(digest: str) -> Dict[str, Any]:
    """Load a schema document by digest."""

    path = SCHEMA_DIR / f"{digest}.json"
    if not path.exists():
        raise FileNotFoundError(digest)
    return json.loads(path.read_text())


def save_code(code: str) -> str:
    """Persist generated reference code and return its digest."""

    _ensure_dirs()
    payload = code.encode()
    digest = _stable_digest(payload)
    (CODE_DIR / f"{digest}.py").write_text(code)
    return digest


def save_manifest(manifest: Dict[str, Any]) -> str:
    """Persist a manifest dictionary and append it to the manifest index."""

    _ensure_dirs()
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = _stable_digest(payload)
    _write_json(MANIFEST_DIR / f"{digest}.json", manifest)

    index_entry = {
        "sha": digest,
        "canonical": manifest.get("canonical"),
        "domain": manifest.get("domain"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with MANIFEST_INDEX.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_entry, sort_keys=True) + "\n")

    return digest


def load_manifest(digest: str) -> Dict[str, Any]:
    """Load a manifest document by digest."""

    path = MANIFEST_DIR / f"{digest}.json"
    if not path.exists():
        raise FileNotFoundError(digest)
    return json.loads(path.read_text())


def save_manifest_entry(
    canonical: str, domain: str, manifest: Dict[str, Any]
) -> str:  # pragma: no cover - legacy shim
    """Deprecated helper retained for backwards compatibility."""

    manifest = dict(manifest)
    manifest.setdefault("canonical", canonical)
    manifest.setdefault("domain", domain)
    return save_manifest(manifest)


def iter_manifest_index(domain: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """Yield entries from the manifest index, optionally filtered by domain."""

    if not MANIFEST_INDEX.exists():
        return []

    with MANIFEST_INDEX.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    entries: list[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if domain is None or entry.get("domain") == domain:
            entries.append(entry)
    return entries


def latest_manifest_hash(domain: Optional[str] = None) -> Optional[str]:
    """Return the newest manifest hash for the given domain, if any."""

    entries = list(iter_manifest_index(domain))
    if not entries:
        return None
    entries = sorted(entries, key=lambda e: e.get("created_at", ""))
    return entries[-1].get("sha")
