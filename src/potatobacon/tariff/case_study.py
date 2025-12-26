from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from fastapi.testclient import TestClient

from potatobacon.proofs.canonical import canonical_json, compute_payload_hash
from potatobacon.proofs.store import get_default_store
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID
from potatobacon.tariff.e2e_runner import ProofReplayResult
from potatobacon.tariff.optimizer import compute_net_savings_projection
from potatobacon.tariff.suggest import DOCUMENTATION_LEVER_ID
from potatobacon.tariff.sku_models import FactOverrideModel
from potatobacon.tariff.context_loader import load_atoms_for_context
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.law.solver_z3 import analyze_scenario


CASE_STUDY_SKU = {
    "sku_id": "RW-ELEC-USB-CABLE-CASE-001",
    "description": "USB-C braided cable assembly with connector pair and copper conductors",
    "origin_country": "VN",
    "declared_value_per_unit": 3.2,
    "annual_volume": 120000,
}

REFINE_PLAN: Dict[str, Dict[str, Any]] = {
    "electronics_insulated_conductors": {
        "value": True,
        "source": "supplier_insulation_spec",
        "confidence": 0.92,
        "evidence_ids": [],
    }
}

EVIDENCE_BLOB = b"supplier_insulation_certificate: PVC jacketed conductor evidence"
EVIDENCE_FILENAME = "insulation_certificate.json"
EVIDENCE_CONTENT_TYPE = "application/json"


@dataclass(frozen=True)
class CaseStudyOutputs:
    json_path: Path
    markdown_path: Path
    audit_pack_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _timestamp_label(timestamp: str | None = None) -> str:
    if timestamp:
        return timestamp
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_new_sku_id(sku_id: str) -> None:
    smoke_path = _repo_root() / "data" / "realworld_smoke_skus.jsonl"
    if not smoke_path.exists():
        return
    for line in smoke_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("sku_id") == sku_id:
                raise ValueError(f"SKU {sku_id} already present in realworld_smoke_skus.jsonl")
        except json.JSONDecodeError:
            continue


def _case_output_paths(output_dir: Path, timestamp: str, sku_id: str) -> CaseStudyOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_sku = sku_id.replace("/", "-")
    base = f"{timestamp}_{safe_sku}"
    return CaseStudyOutputs(
        json_path=output_dir / f"{base}.json",
        markdown_path=output_dir / f"{base}.md",
        audit_pack_path=output_dir / f"{base}_audit-pack.pdf",
    )


def _upload_evidence(client: TestClient) -> str:
    response = client.post(
        "/api/tariff/evidence/upload",
        files={"file": (EVIDENCE_FILENAME, EVIDENCE_BLOB, EVIDENCE_CONTENT_TYPE)},
    )
    if response.status_code != 200:
        raise RuntimeError(f"evidence upload failed: {response.status_code} {response.text}")
    return response.json()["evidence_id"]


def _select_refine_overrides(
    conditional_pathways: Iterable[Mapping[str, Any]],
    baseline_duty: float | None,
    evidence_id: str,
) -> Dict[str, FactOverrideModel]:
    viable: List[Mapping[str, Any]] = []
    for pathway in conditional_pathways:
        missing = pathway.get("missing_facts") or []
        if not missing:
            continue
        if not set(missing).issubset(REFINE_PLAN.keys()):
            continue
        if baseline_duty is not None and pathway.get("duty_rate") is not None:
            if float(pathway["duty_rate"]) >= float(baseline_duty):
                continue
        viable.append(pathway)

    if not viable:
        raise ValueError("No conditional pathways match the refine plan for a cheaper lane")

    viable.sort(
        key=lambda path: (
            len(path.get("missing_facts") or []),
            float(path.get("duty_rate") or 999.0),
            path.get("atom_id", ""),
        )
    )
    selected = viable[0]
    overrides: Dict[str, FactOverrideModel] = {}
    for fact_key in sorted(selected.get("missing_facts") or []):
        plan = dict(REFINE_PLAN[fact_key])
        plan["evidence_ids"] = [evidence_id]
        overrides[fact_key] = FactOverrideModel(**plan)
    return overrides


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_proof_replay(proof_id: str | None, payload_hash: str | None, law_context: str) -> ProofReplayResult:
    if not proof_id:
        return ProofReplayResult(False, "missing proof_id")
    store = get_default_store()
    record = store.get_proof(proof_id)
    if not record:
        return ProofReplayResult(False, "proof not found")
    computed_hash = compute_payload_hash(record)
    if payload_hash and computed_hash != payload_hash:
        return ProofReplayResult(False, "payload hash mismatch")

    context = record.get("law_context") or law_context or DEFAULT_CONTEXT_ID
    atoms, _ = load_atoms_for_context(context)
    compiled = record.get("compiled_facts", {})
    baseline_facts = compiled.get("baseline") or {}
    optimized_facts = compiled.get("optimized") or {}

    baseline_sat, active_baseline, _ = analyze_scenario(baseline_facts, atoms)
    optimized_sat, active_opt, _ = analyze_scenario(optimized_facts, atoms)
    solver_status = "SAT" if baseline_sat and optimized_sat else "UNSAT"
    if solver_status != record.get("solver_result"):
        return ProofReplayResult(False, "solver status mismatch")

    def _duty_codes(active_atoms: Iterable[Any]) -> List[str]:
        codes: List[str] = []
        for atom in active_atoms:
            source = atom.source_id if hasattr(atom, "source_id") else atom.get("source_id") if isinstance(atom, dict) else None
            if source and source in DUTY_RATES:
                codes.append(source)
        return sorted(codes)

    recorded_baseline = _duty_codes(record.get("baseline", {}).get("active_atoms", []))
    recorded_optimized = _duty_codes(record.get("optimized", {}).get("active_atoms", []))
    replay_baseline = _duty_codes(active_baseline)
    replay_optimized = _duty_codes(active_opt)

    if recorded_baseline != replay_baseline or recorded_optimized != replay_optimized:
        return ProofReplayResult(False, "active duty codes diverged")

    def _rate_and_status(active_atoms: Iterable[Any], sat: bool) -> tuple[float | None, str]:
        duty_atoms = [atom for atom in active_atoms if atom.source_id in DUTY_RATES]
        if not sat:
            return None, "UNSAT"
        if not duty_atoms:
            return None, "NO_DUTY_RULE_ACTIVE"
        ranked = sorted(
            duty_atoms,
            key=lambda atom: (
                float(DUTY_RATES.get(atom.source_id, 999.0)),
                -len(getattr(atom, "guard", []) or []),
                atom.source_id,
            ),
        )
        return float(DUTY_RATES[ranked[0].source_id]), "OK"

    baseline_rate_calc, baseline_status_calc = _rate_and_status(active_baseline, baseline_sat)
    optimized_rate_calc, optimized_status_calc = _rate_and_status(active_opt, optimized_sat)

    recorded_baseline_status = record.get("baseline", {}).get("duty_status") or (
        "NO_DUTY_RULE_ACTIVE" if record.get("baseline", {}).get("duty_rate") is None else "OK"
    )
    recorded_optimized_status = record.get("optimized", {}).get("duty_status") or (
        "NO_DUTY_RULE_ACTIVE" if record.get("optimized", {}).get("duty_rate") is None else "OK"
    )

    if baseline_status_calc != recorded_baseline_status:
        return ProofReplayResult(False, "baseline duty status mismatch")
    if optimized_status_calc != recorded_optimized_status:
        return ProofReplayResult(False, "optimized duty status mismatch")

    if baseline_status_calc == "OK" and baseline_rate_calc is not None:
        recorded_baseline_rate = record.get("baseline", {}).get("duty_rate")
        if recorded_baseline_rate not in (None, baseline_rate_calc) and abs(
            recorded_baseline_rate - baseline_rate_calc
        ) > 1e-6:
            return ProofReplayResult(False, "baseline duty mismatch")

    if optimized_status_calc == "OK" and optimized_rate_calc is not None:
        recorded_optimized_rate = record.get("optimized", {}).get("duty_rate")
        if recorded_optimized_rate not in (None, optimized_rate_calc) and abs(
            recorded_optimized_rate - optimized_rate_calc
        ) > 1e-6:
            return ProofReplayResult(False, "optimized duty mismatch")

    return ProofReplayResult(True, "ok")


def run_case_study(
    client: TestClient,
    *,
    seed: int = 2025,
    output_dir: Path | None = None,
    timestamp: str | None = None,
) -> Dict[str, Any]:
    random.seed(seed)
    _ensure_new_sku_id(CASE_STUDY_SKU["sku_id"])

    output_dir = output_dir or Path("reports") / "case_studies"
    stamp = _timestamp_label(timestamp)
    output_paths = _case_output_paths(output_dir, stamp, CASE_STUDY_SKU["sku_id"])

    create_resp = client.post("/api/tariff/skus", json=CASE_STUDY_SKU)
    if create_resp.status_code != 200:
        raise RuntimeError(f"failed to create sku: {create_resp.status_code} {create_resp.text}")

    baseline_resp = client.post(
        f"/api/tariff/skus/{CASE_STUDY_SKU['sku_id']}/dossier",
        json={"law_context": DEFAULT_CONTEXT_ID, "optimize": True},
    )
    if baseline_resp.status_code != 200:
        raise RuntimeError(f"baseline dossier failed: {baseline_resp.status_code} {baseline_resp.text}")
    baseline_body = baseline_resp.json()

    suggest_payload = {
        "sku_id": CASE_STUDY_SKU["sku_id"],
        "description": CASE_STUDY_SKU["description"],
        "origin_country": CASE_STUDY_SKU["origin_country"],
        "declared_value_per_unit": CASE_STUDY_SKU["declared_value_per_unit"],
        "annual_volume": CASE_STUDY_SKU["annual_volume"],
        "law_context": DEFAULT_CONTEXT_ID,
        "top_k": 5,
        "seed": seed,
    }
    suggest_resp = client.post("/api/tariff/suggest", json=suggest_payload)
    if suggest_resp.status_code != 200:
        raise RuntimeError(f"suggest failed: {suggest_resp.status_code} {suggest_resp.text}")
    suggest_body = suggest_resp.json()

    documentation_levers = sorted(
        [
            item.get("lever_id")
            for item in suggest_body.get("suggestions", [])
            if item.get("lever_category") == DOCUMENTATION_LEVER_ID
        ]
    )

    baseline_assigned = baseline_body.get("baseline_assigned") or {}
    baseline_candidates = sorted(
        baseline_body.get("baseline", {}).get("candidates", []),
        key=lambda item: item.get("candidate_id", ""),
    )
    missing_facts = sorted(baseline_body.get("questions", {}).get("missing_facts", []))
    optimized_suggestion = (baseline_body.get("optimized") or {}).get("suggestion") or {}
    overlays = optimized_suggestion.get("overlays")

    baseline_proof = {
        "proof_id": baseline_body.get("proof_id"),
        "proof_payload_hash": baseline_body.get("proof_payload_hash"),
    }
    baseline_replay = _validate_proof_replay(
        baseline_proof["proof_id"],
        baseline_proof["proof_payload_hash"],
        baseline_body.get("law_context") or DEFAULT_CONTEXT_ID,
    )

    print(
        canonical_json(
            {
                "phase": "baseline",
                "baseline_assigned": baseline_assigned,
                "baseline_candidates": baseline_candidates,
                "missing_facts": missing_facts,
                "documentation_levers": documentation_levers,
                "overlays": overlays or {},
                "baseline_effective_duty_rate": optimized_suggestion.get("baseline_effective_duty_rate"),
                "proof_id": baseline_proof["proof_id"],
                "proof_replay": "PASS" if baseline_replay.ok else "FAIL",
            }
        )
    )

    session_resp = client.post(f"/api/tariff/skus/{CASE_STUDY_SKU['sku_id']}/sessions", json={})
    if session_resp.status_code != 200:
        raise RuntimeError(f"session creation failed: {session_resp.status_code} {session_resp.text}")
    session_id = session_resp.json()["session_id"]

    evidence_id = _upload_evidence(client)
    refine_overrides = _select_refine_overrides(
        baseline_body.get("conditional_pathways", []),
        baseline_assigned.get("duty_rate"),
        evidence_id,
    )

    refine_payload = {
        "attached_evidence_ids": [evidence_id],
        "fact_overrides": {key: override.model_dump() for key, override in refine_overrides.items()},
        "optimize": True,
        "evidence_requested": True,
    }
    refine_resp = client.post(f"/api/tariff/sessions/{session_id}/refine", json=refine_payload)
    if refine_resp.status_code != 200:
        raise RuntimeError(f"refine failed: {refine_resp.status_code} {refine_resp.text}")
    refine_body = refine_resp.json()["dossier"]

    refined_suggestion = (refine_body.get("optimized") or {}).get("suggestion") or {}
    refined_assigned = refine_body.get("baseline_assigned") or {}
    duty_delta = None
    if baseline_assigned.get("duty_rate") is not None and refined_assigned.get("duty_rate") is not None:
        duty_delta = float(baseline_assigned["duty_rate"]) - float(refined_assigned["duty_rate"])

    fact_deltas = []
    baseline_facts = (baseline_body.get("compiled_facts") or {}).get("normalized") or {}
    refined_facts = (refine_body.get("compiled_facts") or {}).get("normalized") or {}
    for key in sorted(refine_overrides.keys()):
        fact_deltas.append({"fact_key": key, "before": baseline_facts.get(key), "after": refined_facts.get(key)})

    net_savings = refined_suggestion.get("net_savings")
    if not net_savings:
        net_savings_model = compute_net_savings_projection(
            baseline_rate=baseline_assigned.get("duty_rate"),
            optimized_rate=refined_assigned.get("duty_rate"),
            declared_value_per_unit=CASE_STUDY_SKU["declared_value_per_unit"],
            annual_volume=CASE_STUDY_SKU["annual_volume"],
        )
        net_savings = net_savings_model.model_dump()

    net_annual_savings = net_savings.get("net_annual_savings")
    refine_status = refine_body.get("status")
    if refine_status != "OK_OPTIMIZED" and duty_delta and duty_delta > 0:
        refine_status = "OK_CONDITIONAL_OPTIMIZATION"

    refine_proof = {
        "proof_id": refine_body.get("proof_id"),
        "proof_payload_hash": refine_body.get("proof_payload_hash"),
    }
    refine_replay = _validate_proof_replay(
        refine_proof["proof_id"],
        refine_proof["proof_payload_hash"],
        refine_body.get("law_context") or DEFAULT_CONTEXT_ID,
    )

    print(
        canonical_json(
            {
                "phase": "refine",
                "chosen_lever": {
                    "lever_id": refined_suggestion.get("lever_id"),
                    "lever_category": refined_suggestion.get("lever_category") or "PHYSICAL",
                },
                "fact_deltas": fact_deltas,
                "duty_delta": duty_delta,
                "net_savings": net_savings,
                "payback_months": net_savings.get("payback_months"),
                "ranking_score": refined_suggestion.get("ranking_score"),
                "proof_id": refine_proof["proof_id"],
                "proof_replay": "PASS" if refine_replay.ok else "FAIL",
            }
        )
    )

    audit_resp = client.get(f"/api/tariff/proofs/{refine_proof['proof_id']}/audit-pack")
    if audit_resp.status_code != 200:
        raise RuntimeError(f"audit pack fetch failed: {audit_resp.status_code} {audit_resp.text}")
    output_paths.audit_pack_path.write_bytes(audit_resp.content)

    inputs_hashes = {
        "sku_payload_hash": compute_payload_hash(CASE_STUDY_SKU),
        "refine_plan_hash": compute_payload_hash(REFINE_PLAN),
        "evidence_payload_hash": _hash_bytes(EVIDENCE_BLOB),
        "refine_payload_hash": compute_payload_hash(refine_payload),
    }

    proof_store = get_default_store()
    baseline_record = proof_store.get_proof(baseline_proof["proof_id"]) if baseline_proof["proof_id"] else None
    refine_record = proof_store.get_proof(refine_proof["proof_id"]) if refine_proof["proof_id"] else None

    case_report = {
        "case_id": f"{stamp}_{CASE_STUDY_SKU['sku_id']}",
        "timestamp": stamp,
        "sku_id": CASE_STUDY_SKU["sku_id"],
        "inputs": inputs_hashes,
        "baseline": {
            "status": baseline_body.get("status"),
            "baseline_assigned": baseline_assigned,
            "baseline_candidates": baseline_candidates,
            "missing_facts": missing_facts,
            "documentation_levers": documentation_levers,
            "overlays": overlays or {},
            "baseline_effective_duty_rate": optimized_suggestion.get("baseline_effective_duty_rate"),
            "proof": {
                **baseline_proof,
                "record_hash": compute_payload_hash(baseline_record) if baseline_record else None,
            },
            "proof_replay": {"ok": baseline_replay.ok, "message": baseline_replay.message},
        },
        "refine": {
            "status": refine_status,
            "dossier_status": refine_body.get("status"),
            "fact_overrides": {key: override.serializable_dict() for key, override in refine_overrides.items()},
            "evidence_id": evidence_id,
            "chosen_lever": {
                "lever_id": refined_suggestion.get("lever_id"),
                "lever_category": refined_suggestion.get("lever_category"),
            },
            "fact_deltas": fact_deltas,
            "duty_delta": duty_delta,
            "net_savings": net_savings,
            "payback_months": net_savings.get("payback_months"),
            "ranking_score": refined_suggestion.get("ranking_score"),
            "proof": {
                **refine_proof,
                "record_hash": compute_payload_hash(refine_record) if refine_record else None,
            },
            "proof_replay": {"ok": refine_replay.ok, "message": refine_replay.message},
        },
        "audit_pack": {
            "path": str(output_paths.audit_pack_path),
            "bytes": output_paths.audit_pack_path.stat().st_size,
        },
    }

    output_paths.json_path.write_text(canonical_json(case_report), encoding="utf-8")
    markdown_lines = [
        "# CALE-TARIFF Case Study",
        "",
        f"- Case ID: {case_report['case_id']}",
        f"- SKU: {CASE_STUDY_SKU['sku_id']}",
        f"- Baseline status: {case_report['baseline']['status']}",
        f"- Refine status: {case_report['refine']['status']}",
        f"- Net annual savings: {net_annual_savings}",
        f"- Proof replay: {case_report['refine']['proof_replay']['ok']}",
        f"- Audit pack: {output_paths.audit_pack_path.name}",
    ]
    output_paths.markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    return {
        "baseline": case_report["baseline"],
        "refine": case_report["refine"],
        "audit_pack": case_report["audit_pack"],
        "report_paths": {
            "json": str(output_paths.json_path),
            "markdown": str(output_paths.markdown_path),
        },
    }
