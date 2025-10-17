from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import json
import hashlib
import platform
import sympy as sp

ART_ROOT = Path("artifacts")
ART_MANIFEST = ART_ROOT / "manifests"
ART_CODE = ART_ROOT / "code"

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
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def stable_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

def ensure_dirs():
    ART_MANIFEST.mkdir(parents=True, exist_ok=True)
    ART_CODE.mkdir(parents=True, exist_ok=True)

def persist_manifest(man: ComputationManifest) -> str:
    ensure_dirs()
    h = man.stable_hash()
    (ART_MANIFEST / f"{h}.json").write_text(man.to_json())
    return h

def persist_code(code_str: str) -> str:
    ensure_dirs()
    digest = hashlib.sha256(code_str.encode()).hexdigest()
    (ART_CODE / f"{digest}.py").write_text(code_str)
    return digest

def load_manifest(hash_value: str) -> Dict[str, Any]:
    p = ART_MANIFEST / f"{hash_value}.json"
    if not p.exists():
        raise FileNotFoundError(hash_value)
    return json.loads(p.read_text())
