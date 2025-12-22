import pytest

from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID
from potatobacon.tariff.audit_pack import generate_audit_pack_pdf


def test_generate_audit_pack_pdf_contains_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    handle = record_tariff_proof(
        law_context=DEFAULT_CONTEXT_ID,
        base_facts={"electronics_cable_or_connector": True},
        mutations={},
        baseline_active=[],
        optimized_active=[],
        baseline_sat=True,
        optimized_sat=True,
        baseline_duty_rate=2.0,
        optimized_duty_rate=2.0,
        baseline_scenario={"electronics_cable_or_connector": True},
        optimized_scenario={"electronics_cable_or_connector": True},
        baseline_duty_status="OK",
        optimized_duty_status="OK",
        baseline_unsat_core=[],
        optimized_unsat_core=[],
        provenance_chain=[],
        overlays={"baseline": [{"overlay_name": "Overlay Demo", "additional_rate": 5.0, "applies": True, "reason": "demo", "requires_review": True, "stop_optimization": False}], "optimized": []},
        tariff_manifest_hash="HASH",
    )

    pdf_bytes = generate_audit_pack_pdf(handle.proof_id)
    assert len(pdf_bytes) > 800
    assert f"Proof: {handle.proof_id}".encode("utf-8") in pdf_bytes
    assert f"Law context: {DEFAULT_CONTEXT_ID}".encode("utf-8") in pdf_bytes
    assert b"Overlay Demo" in pdf_bytes
