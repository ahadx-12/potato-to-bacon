import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_audit_pack_endpoint_returns_pdf(system_client: TestClient):
    sku_id = "SKU-AUDIT-PACK"
    sku_payload = {
        "sku_id": sku_id,
        "description": "Electronics connector harness with braided shield",
        "declared_value_per_unit": 12.0,
        "annual_volume": 1000,
    }
    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text

    dossier_resp = system_client.post(f"/api/tariff/skus/{sku_id}/dossier", json={"optimize": True})
    assert dossier_resp.status_code == 200, dossier_resp.text
    proof_id = dossier_resp.json()["proof_id"]

    audit_resp = system_client.get(f"/api/tariff/proofs/{proof_id}/audit-pack")
    assert audit_resp.status_code == 200, audit_resp.text
    assert audit_resp.headers["content-type"] == "application/pdf"
    assert len(audit_resp.content) > 500
