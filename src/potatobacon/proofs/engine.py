from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.proofs.canonical import canonical_json, compute_payload_hash

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


def _atom_sort_key(atom: Dict[str, Any]) -> tuple[str, ...]:
    if "atom_id" in atom:
        return (
            str(atom.get("source_id", "")),
            str(atom.get("atom_id", "")),
            str(atom.get("section", "")),
            str(atom.get("text", "")),
        )
    return (
        str(atom.get("source_id", "")),
        str(atom.get("section", "")),
        str(atom.get("text", "")),
    )


def _sorted_atoms(atoms: Iterable[PolicyAtom]) -> List[Dict[str, Any]]:
    serialized = [_serialize_atom(atom) for atom in atoms]
    serialized.sort(key=_atom_sort_key)
    return serialized


def _sorted_provenance(chain: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries = [dict(item) for item in chain]
    entries.sort(key=_atom_sort_key)
    return entries


def _sorted_evidence_pack(pack: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not pack:
        return pack
    sorted_pack = {**pack}
    fact_evidence = sorted_pack.get("fact_evidence")
    if isinstance(fact_evidence, list):
        sorted_fact_evidence: list[Dict[str, Any]] = []
        for item in fact_evidence:
            normalized_item = dict(item)
            evidence_list = normalized_item.get("evidence")
            if isinstance(evidence_list, list):
                normalized_item["evidence"] = sorted(
                    evidence_list,
                    key=lambda ev: (
                        ev.get("source", ""),
                        ev.get("start", -1),
                        ev.get("end", -1),
                        ev.get("snippet", ""),
                    ),
                )
            sorted_fact_evidence.append(normalized_item)

        sorted_fact_evidence.sort(
            key=lambda item: (
                item.get("fact_key", ""),
                str(item.get("value")),
                len(item.get("evidence", [])),
            )
        )
        sorted_pack["fact_evidence"] = sorted_fact_evidence
    extraction = sorted_pack.get("extraction_evidence")
    if isinstance(extraction, list):
        sorted_pack["extraction_evidence"] = sorted(
            extraction,
            key=lambda item: (
                item.get("source", ""),
                item.get("start", -1),
                item.get("end", -1),
                item.get("snippet", ""),
            ),
        )
    metadata = sorted_pack.get("sku_metadata")
    if isinstance(metadata, dict):
        sorted_pack["sku_metadata"] = {key: metadata[key] for key in sorted(metadata.keys())}
    return sorted_pack


@dataclass(frozen=True)
class ProofHandle:
    proof_id: str
    proof_payload_hash: str


def build_proof_id(proof_material: Dict[str, Any]) -> str:
    material_json = canonical_json(proof_material)
    return sha256(material_json.encode("utf-8")).hexdigest()


def record_tariff_proof(
    *,
    law_context: str,
    base_facts: Dict[str, Any],
    mutations: Dict[str, Any] | None,
    baseline_active: Iterable[PolicyAtom],
    optimized_active: Iterable[PolicyAtom],
    baseline_sat: bool,
    optimized_sat: bool,
    baseline_duty_rate: float | None,
    optimized_duty_rate: float | None,
    baseline_duty_status: str = "OK",
    optimized_duty_status: str = "OK",
    baseline_scenario: Dict[str, Any] | None = None,
    optimized_scenario: Dict[str, Any] | None = None,
    baseline_unsat_core: List[PolicyAtom] | None = None,
    optimized_unsat_core: List[PolicyAtom] | None = None,
    provenance_chain: Iterable[Dict[str, Any]] | None = None,
    evidence_pack: Dict[str, Any] | None = None,
    store: ProofStore | None = None,
    tariff_manifest_hash: str | None = None,
) -> ProofHandle:
    """Persist a tariff proof and return a reference handle."""

    timestamp = datetime.now(timezone.utc).isoformat()
    ordered_evidence_pack = _sorted_evidence_pack(evidence_pack)
    proof_material: Dict[str, Any] = {
        "timestamp": timestamp,
        "law_context": law_context,
        "input": {"scenario": deepcopy(base_facts), "mutations": deepcopy(mutations or {})},
        "solver_result": "SAT" if baseline_sat and optimized_sat else "UNSAT",
        "baseline": {
            "active_atoms": _sorted_atoms(baseline_active),
            "unsat_core": _sorted_atoms(baseline_unsat_core or []),
            "sat": baseline_sat,
            "duty_rate": baseline_duty_rate,
            "duty_status": baseline_duty_status,
        },
        "optimized": {
            "active_atoms": _sorted_atoms(optimized_active),
            "unsat_core": _sorted_atoms(optimized_unsat_core or []),
            "sat": optimized_sat,
            "duty_rate": optimized_duty_rate,
            "duty_status": optimized_duty_status,
        },
        "compiled_facts": {
            "baseline": deepcopy(baseline_scenario or base_facts),
            "optimized": deepcopy(optimized_scenario or base_facts),
        },
    }

    if tariff_manifest_hash:
        proof_material["tariff_manifest_hash"] = tariff_manifest_hash

    if ordered_evidence_pack:
        proof_material["evidence_pack"] = ordered_evidence_pack

    if provenance_chain is not None:
        proof_material["provenance_chain"] = _sorted_provenance(provenance_chain)

    proof_payload_hash = compute_payload_hash(proof_material)
    proof_id = build_proof_id(proof_material)
    proof_record = {
        "proof_id": proof_id,
        "proof_payload_hash": proof_payload_hash,
        **proof_material,
    }
    proof_store = store or get_default_store()
    proof_store.save_proof(proof_record)
    return ProofHandle(proof_id=proof_id, proof_payload_hash=proof_payload_hash)
