from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.atom_utils import duty_rate_index
from potatobacon.tariff.gri_atoms import build_gri_atoms, gri_text_hash

CONTEXTS_DIR = Path(__file__).resolve().parent / "contexts"
MANIFESTS_DIR = CONTEXTS_DIR / "manifests"
RULES_DIR = CONTEXTS_DIR / "rules"
DEFAULT_CONTEXT_ID = "HTS_US_2025_SLICE"
REPO_ROOT = Path(__file__).resolve().parents[3]

_atoms_cache: Dict[str, Tuple[List[PolicyAtom], Dict[str, Any]]] = {}


def _normalized_metadata(metadata: Any) -> Any:
    if metadata is None:
        return None
    return json.loads(json.dumps(metadata, sort_keys=True, separators=(",", ":")))


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _resolve_source_path(source: str) -> Path:
    path = Path(source)
    if path.is_absolute():
        return path
    candidate = CONTEXTS_DIR / path
    if candidate.exists():
        return candidate
    return REPO_ROOT / path


def _section_notes_hash(manifest: Dict[str, Any]) -> str | None:
    sources = manifest.get("sources") or []
    note_sources = [src for src in sources if "note" in str(src).lower()]
    if note_sources:
        hashes = []
        for src in sorted(note_sources):
            path = _resolve_source_path(str(src))
            if path.exists():
                hashes.append(_hash_file(path))
        if hashes:
            return _hash_bytes("".join(hashes).encode("utf-8"))

    loader = manifest.get("loader") or {}
    if loader.get("type") == "json_rules":
        rules_file = loader.get("rules_file")
        if rules_file:
            rules_path = _resolve_source_path(str(rules_file))
            if rules_path.exists():
                return _hash_file(rules_path)
    return None


def _apply_reference_id(
    *,
    revision_id: str,
    citation: Dict[str, Any],
    fallback_note_id: str,
) -> Dict[str, Any]:
    chapter = citation.get("chapter") or "00"
    note_id = citation.get("note_id") or fallback_note_id
    reference_id = f"{revision_id}::{chapter}::{note_id}"
    citation["revision_id"] = revision_id
    citation["reference_id"] = reference_id
    return citation


def list_context_manifests(domain: str = "tariff") -> List[Dict[str, Any]]:
    """Return all manifests for the requested domain sorted by ``context_id``."""

    manifests: list[Dict[str, Any]] = []
    if not MANIFESTS_DIR.exists():
        return manifests

    for manifest_path in MANIFESTS_DIR.glob("*.json"):
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("domain") == domain:
            manifests.append(manifest)

    manifests.sort(key=lambda manifest: manifest.get("context_id", ""))
    return manifests


def available_context_ids(domain: str = "tariff") -> List[str]:
    """Return sorted context identifiers for the domain."""

    return [manifest["context_id"] for manifest in list_context_manifests(domain=domain)]


def get_context_manifest(context_id: str) -> Dict[str, Any]:
    """Load a manifest by ``context_id`` raising ``KeyError`` when missing."""

    manifest_path = MANIFESTS_DIR / f"{context_id}.json"
    if not manifest_path.exists():
        raise KeyError(context_id)

    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atom_to_dict(atom: PolicyAtom) -> Dict[str, Any]:
    return {
        "guard": list(atom.guard),
        "outcome": dict(atom.outcome),
        "source_id": atom.source_id,
        "statute": atom.statute,
        "section": atom.section,
        "text": atom.text,
        "modality": atom.modality,
        "action": atom.action,
        "rule_type": atom.rule_type,
        "atom_id": atom.atom_id,
        "metadata": _normalized_metadata(getattr(atom, "metadata", None)),
    }


def compute_context_hash(manifest: Dict[str, Any], atoms: Iterable[PolicyAtom]) -> str:
    """Compute a deterministic hash for a manifest and its atoms."""

    manifest_blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    atoms_blob = json.dumps(
        [_atom_to_dict(atom) for atom in atoms],
        sort_keys=True,
        separators=(",", ":"),
    )
    hasher = hashlib.sha256()
    hasher.update(manifest_blob.encode("utf-8"))
    hasher.update(atoms_blob.encode("utf-8"))
    return hasher.hexdigest()


def _load_python_atoms(loader_spec: Dict[str, Any]) -> List[PolicyAtom]:
    callable_path = loader_spec.get("callable")
    if not callable_path or ":" not in callable_path:
        raise ValueError("Invalid python loader spec")

    module_path, func_name = callable_path.split(":", 1)
    module = importlib.import_module(module_path)
    loader_callable = getattr(module, func_name, None)
    if not callable(loader_callable):
        raise ValueError(f"Loader callable not found: {callable_path}")

    atoms = loader_callable()
    if not isinstance(atoms, list):
        raise ValueError("Loader callable must return a list of PolicyAtom")
    return atoms


def _rule_obj_to_atom(rule: Dict[str, Any]) -> PolicyAtom:
    guard = list(rule.get("guard") or [])
    outcome = dict(rule.get("outcome") or {})
    return PolicyAtom(
        guard=guard,
        outcome=outcome,
        source_id=rule.get("source_id", ""),
        statute=rule.get("statute", ""),
        section=rule.get("section", ""),
        text=rule.get("text", ""),
        modality=rule.get("modality", outcome.get("modality", "")),
        action=rule.get("action", outcome.get("action", "")),
        rule_type=rule.get("rule_type", "STATUTE"),
        atom_id=rule.get("atom_id"),
        metadata=rule.get("metadata"),
    )


def _load_json_rule_atoms(loader_spec: Dict[str, Any]) -> List[PolicyAtom]:
    rules_file = loader_spec.get("rules_file")
    if not rules_file:
        raise ValueError("json_rules loader requires rules_file")

    rules_path = Path(rules_file)
    if not rules_path.is_absolute():
        rules_path = CONTEXTS_DIR / rules_file
    if not rules_path.exists():
        raise ValueError(f"Rules file not found: {rules_path}")

    with rules_path.open("r", encoding="utf-8") as handle:
        raw_rules = json.load(handle)

    atoms: list[PolicyAtom] = []
    for rule in raw_rules:
        atom = _rule_obj_to_atom(rule)
        atoms.append(atom)
    return atoms


def _validate_atoms(atoms: List[PolicyAtom]) -> None:
    if not isinstance(atoms, list) or not all(isinstance(atom, PolicyAtom) for atom in atoms):
        raise ValueError("Loader did not return PolicyAtom list")


def load_atoms_for_context(context_id: str) -> Tuple[List[PolicyAtom], Dict[str, Any]]:
    """Load atoms and metadata for ``context_id`` with in-process caching."""

    if context_id in _atoms_cache:
        cached_atoms, cached_meta = _atoms_cache[context_id]
        return list(cached_atoms), dict(cached_meta)

    manifest = get_context_manifest(context_id)
    revision_id = manifest.get("revision_id", manifest.get("context_id", context_id))
    loader_spec = manifest.get("loader", {})
    loader_type = loader_spec.get("type")

    if loader_type == "python":
        atoms = _load_python_atoms(loader_spec)
    elif loader_type == "json_rules":
        atoms = _load_json_rule_atoms(loader_spec)
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}")

    _validate_atoms(atoms)

    gri_atoms = build_gri_atoms(revision_id=revision_id)
    atoms = list(gri_atoms) + list(atoms)

    for atom in atoms:
        metadata = getattr(atom, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        citation = metadata.get("citation")
        if isinstance(citation, dict):
            metadata["citation"] = _apply_reference_id(
                revision_id=revision_id,
                citation=dict(citation),
                fallback_note_id=atom.source_id,
            )
            atom.metadata = metadata

    manifest_hash = compute_context_hash(manifest, atoms)
    duty_rates = duty_rate_index(atoms)
    section_notes_hash = manifest.get("section_notes_hash") or _section_notes_hash(manifest)
    gri_hash = manifest.get("gri_text_hash") or gri_text_hash()
    metadata = {
        "context_id": manifest.get("context_id", context_id),
        "domain": manifest.get("domain", "tariff"),
        "jurisdiction": manifest.get("jurisdiction"),
        "effective_from": manifest.get("effective_from"),
        "effective_to": manifest.get("effective_to"),
        "description": manifest.get("description"),
        "section_notes_hash": section_notes_hash,
        "gri_text_hash": gri_hash,
        "revision_id": revision_id,
        "manifest_hash": manifest_hash,
        "atoms_count": len(atoms),
        "duty_rates": duty_rates,
    }

    _atoms_cache[context_id] = (list(atoms), dict(metadata))
    return list(atoms), dict(metadata)
