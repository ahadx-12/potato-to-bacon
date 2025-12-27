import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_origin_engineering_tariff_shift_lane(system_client: TestClient):
    sku_id = "SKU-ORIGIN-TSHIFT"
    sku_payload = {
        "sku_id": sku_id,
        "description": "Insulated cable assembly with copper conductors",
        "origin_country": "MX",
        "declared_value_per_unit": 100.0,
        "annual_volume": 1000,
        "current_hts": "8544.42",
    }
    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text

    bom_csv = """part_name,material,origin_country,hts_code,value,operation
copper wire,copper,CN,7408.19,40,wire drawing
insulation jacket,plastic,CN,3921.90,10,extrusion
cable assembly,assembly,MX,8544.42,50,cable assembly
"""
    upload = system_client.post(
        "/api/tariff/evidence/upload",
        data={"evidence_kind": "bom_csv"},
        files={"file": ("cable_bom.csv", bom_csv.encode("utf-8"), "text/csv")},
    )
    assert upload.status_code == 200, upload.text
    evidence_id = upload.json()["evidence_id"]

    dossier_resp = system_client.post(
        f"/api/tariff/skus/{sku_id}/dossier",
        json={"optimize": True, "attached_evidence_ids": [evidence_id]},
    )
    assert dossier_resp.status_code == 200, dossier_resp.text
    dossier = dossier_resp.json()
    lanes = {lane["lane_id"]: lane for lane in dossier["opportunity_ledger"]["entries"]}
    assert lanes["Substantial Transformation Detected"]["status"] == "AVAILABLE_NOW"


@pytest.mark.usefixtures("system_client")
def test_origin_engineering_rvc_and_audit_pack(system_client: TestClient):
    sku_id = "SKU-ORIGIN-RVC"
    sku_payload = {
        "sku_id": sku_id,
        "description": "Electric motor assembly with mixed-origin components",
        "origin_country": "MX",
        "declared_value_per_unit": 100.0,
        "annual_volume": 500,
        "current_hts": "8501.10",
    }
    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text

    bom_csv = """part_name,material,origin_country,value,is_originating_material,operation
motor core,steel,CN,55,no,core machining
housing,steel,MX,45,yes,final assembly
"""
    upload = system_client.post(
        "/api/tariff/evidence/upload",
        data={"evidence_kind": "bom_csv"},
        files={"file": ("motor_bom.csv", bom_csv.encode("utf-8"), "text/csv")},
    )
    assert upload.status_code == 200, upload.text
    evidence_id = upload.json()["evidence_id"]

    dossier_resp = system_client.post(
        f"/api/tariff/skus/{sku_id}/dossier",
        json={"optimize": True, "attached_evidence_ids": [evidence_id]},
    )
    assert dossier_resp.status_code == 200, dossier_resp.text
    dossier = dossier_resp.json()
    compiled_facts = dossier["compiled_facts"]
    assert compiled_facts["origin_rvc_build_down"] == pytest.approx(45.0)
    lanes = {lane["lane_id"]: lane for lane in dossier["opportunity_ledger"]["entries"]}
    assert lanes["USMCA Qualification"]["status"] == "INELIGIBLE_INSUFFICIENT_RVC"

    proof_id = dossier["proof_id"]
    audit_resp = system_client.get(f"/api/tariff/proofs/{proof_id}/audit-pack")
    assert audit_resp.status_code == 200, audit_resp.text
    assert b"Value-Added Worksheet" in audit_resp.content
