from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import platform
import sympy as sp

from ..storage import load_manifest as storage_load_manifest
from ..storage import save_code as storage_save_code
from ..storage import save_manifest as storage_save_manifest


@dataclass
class ComputationManifest:
    version: str
    canonical: str
    domain: str
    units: Dict[str, str]
    constraints: Dict[str, Any]
    checks_report: Dict[str, Any]
    schema_digest: str
    code_digest: str
    created_by: str = "potato-to-bacon"
    system: str = platform.platform()
    sympy_version: str = sp.__version__

    def to_json(self) -> str:
        import json

        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def stable_hash(self) -> str:
        from hashlib import sha256
        import json

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return sha256(payload).hexdigest()


def persist_manifest(man: ComputationManifest) -> str:
    return storage_save_manifest(asdict(man))


def persist_code(code_str: str) -> str:
    return storage_save_code(code_str)


def load_manifest(hash_value: str) -> Dict[str, Any]:
    return storage_load_manifest(hash_value)
