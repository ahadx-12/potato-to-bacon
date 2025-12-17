"""Canonical proof payload normalization and hashing utilities."""

from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, List

VOLATILE_KEYS = {
    "proof_id",
    "created_at",
    "generated_at",
    "timestamp",
    "run_id",
    "job_id",
    "proof_payload_hash",
}

UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid_like(value: Any) -> bool:
    if isinstance(value, uuid.UUID):
        return True
    if isinstance(value, str) and UUID_PATTERN.match(value):
        return True
    return False


def _atom_sort_key(atom: Dict[str, Any]) -> tuple:
    source = atom.get("source_id", "")
    if "atom_id" in atom:
        return (source, atom.get("atom_id", ""))
    return (source, atom.get("section", ""), atom.get("text", ""))


def _evidence_sort_key(snippet: Dict[str, Any]) -> tuple:
    return (
        snippet.get("source", ""),
        snippet.get("start", -1),
        snippet.get("end", -1),
        snippet.get("snippet", ""),
    )


def _is_atom_dict(entry: Any) -> bool:
    return isinstance(entry, dict) and "source_id" in entry


def _is_evidence_snippet(entry: Any) -> bool:
    return isinstance(entry, dict) and "snippet" in entry


def _is_fact_evidence(entry: Any) -> bool:
    return isinstance(entry, dict) and "fact_key" in entry


def _normalize_list(values: List[Any]) -> List[Any]:
    normalized = [_normalize_value(item) for item in values]

    if all(_is_atom_dict(item) for item in normalized):
        normalized.sort(key=_atom_sort_key)
    elif all(_is_evidence_snippet(item) for item in normalized):
        normalized.sort(key=_evidence_sort_key)
    elif all(_is_fact_evidence(item) for item in normalized):
        normalized.sort(key=lambda item: (item.get("fact_key", ""), json.dumps(item.get("value"), sort_keys=True)))
    return normalized


def _normalize_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in VOLATILE_KEYS:
            continue
        if _is_uuid_like(value) or "uuid" in key:
            continue
        normalized[key] = _normalize_value(value)
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _normalize_dict(value)
    if isinstance(value, list):
        return _normalize_list(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def normalize_for_hash(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied payload stripped of volatile fields with stable ordering."""

    payload_copy = copy.deepcopy(payload)
    return _normalize_value(payload_copy)


def canonical_json(obj: Any) -> str:
    """Serialize *obj* to canonical JSON suitable for hashing."""

    def _default(value: Any):
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value)} is not JSON serializable")

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    """Return a SHA-256 hex digest for the normalized payload."""

    normalized = normalize_for_hash(payload)
    serialized = canonical_json(normalized)
    return sha256(serialized.encode("utf-8")).hexdigest()

