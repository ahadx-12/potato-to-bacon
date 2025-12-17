from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from potatobacon.law.solver_z3 import PolicyAtom

from .store import ProofStore, get_default_store


def _serialize_atom(atom: PolicyAtom) -> Dict[str, Any]:
    return {
        "source_id": atom.source_id,
        "statute": getattr(atom, "statute", ""),
        "section": getattr(atom, "section", ""),
        "text": getattr(atom, "text", ""),
        "jurisdiction": atom.outcome.get("jurisdiction", ""),
        "atom_id": getattr(atom, "atom_id", ""),
    }


def build_proof_id(proof_material: Dict[str, Any]) -> str:
    material_json = json.dumps(proof_material, sort_keys=True)
    return hashlib.sha256(material_json.encode("utf-8")).hexdigest()


def record_tariff_proof(
    *,
    law_context: str,
    base_facts: Dict[str, Any],
    mutations: Dict[str, Any] | None,
    baseline_active: Iterable[PolicyAtom],
    optimized_active: Iterable[PolicyAtom],
    baseline_sat: bool,
    optimized_sat: bool,
    baseline_unsat_core: List[PolicyAtom] | None = None,
    optimized_unsat_core: List[PolicyAtom] | None = None,
    evidence_pack: Dict[str, Any] | None = None,
    store: ProofStore | None = None,
) -> str:
    """Persist a tariff proof and return its proof_id."""

    timestamp = datetime.now(timezone.utc).isoformat()
    proof_material: Dict[str, Any] = {
        "timestamp": timestamp,
        "law_context": law_context,
        "input": {"scenario": base_facts, "mutations": mutations or {}},
        "solver_result": "SAT" if baseline_sat and optimized_sat else "UNSAT",
        "baseline": {
            "active_atoms": [_serialize_atom(atom) for atom in baseline_active],
            "unsat_core": [_serialize_atom(atom) for atom in baseline_unsat_core or []],
            "sat": baseline_sat,
        },
        "optimized": {
            "active_atoms": [_serialize_atom(atom) for atom in optimized_active],
            "unsat_core": [_serialize_atom(atom) for atom in optimized_unsat_core or []],
            "sat": optimized_sat,
        },
    }

    if evidence_pack:
        proof_material["evidence_pack"] = evidence_pack

    proof_id = build_proof_id(proof_material)
    proof_record = {"proof_id": proof_id, **proof_material}
    proof_store = store or get_default_store()
    proof_store.save_proof(proof_record)
    return proof_id
