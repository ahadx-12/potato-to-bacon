import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_bom_csv_evidence_autofills_insulation(system_client: TestClient):
    sku_id = "SKU-EVIDENCE-AUTOFILL"
    sku_payload = {
        "sku_id": sku_id,
        "description": "USB-C cable assembly with copper conductors and molded plugs",
        "declared_value_per_unit": 3.2,
        "annual_volume": 75000,
        "origin_country": "VN",
    }
    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text

    initial = system_client.post(f"/api/tariff/skus/{sku_id}/dossier", json={"optimize": True})
    assert initial.status_code == 200, initial.text
    initial_body = initial.json()
    assert initial_body["baseline_assigned"]["duty_rate"] == pytest.approx(2.0)
    assert any(
        "electronics_insulated_conductors" in path["missing_facts"]
        for path in initial_body["conditional_pathways"]
        if path["atom_id"] == "HTS_ELECTRONICS_SIGNAL_LOW_VOLT"
    )

    bom_csv = """part_name,material,origin_country,value
connector shell,plastic,VN,1.2
copper core,copper,VN,1.5
"""
    upload = system_client.post(
        "/api/tariff/evidence/upload",
        data={"evidence_kind": "bom_csv"},
        files={"file": ("usb_bom.csv", bom_csv.encode("utf-8"), "text/csv")},
    )
    assert upload.status_code == 200, upload.text
    evidence_id = upload.json()["evidence_id"]

    session_resp = system_client.post(f"/api/tariff/skus/{sku_id}/sessions", json={})
    assert session_resp.status_code == 200, session_resp.text
    session_id = session_resp.json()["session_id"]

    refined = system_client.post(
        f"/api/tariff/sessions/{session_id}/refine",
        json={"attached_evidence_ids": [evidence_id], "optimize": True},
    )
    assert refined.status_code == 200, refined.text
    dossier = refined.json()["dossier"]
    assert dossier["analysis_session_id"] == session_id
    assert evidence_id in dossier["attached_evidence_ids"]
    assert dossier["baseline_assigned"]["duty_rate"] == pytest.approx(1.0)
    assert "electronics_insulated_conductors" not in dossier["questions"]["missing_facts"]
