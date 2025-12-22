import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_tariff_sku_dossier_v2_converse(system_client: TestClient):
    sku_id = "SKU-CONVERSE-DEMO"
    sku_payload = {
        "sku_id": sku_id,
        "description": "Canvas sneaker with rubber outsole and textile upper",
        "origin_country": "VN",
        "declared_value_per_unit": 42.0,
        "annual_volume": 5000,
    }
    resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert resp.status_code == 200, resp.text

    dossier_resp = system_client.post(
        f"/api/tariff/skus/{sku_id}/dossier",
        json={"optimize": True, "evidence_requested": True},
    )
    assert dossier_resp.status_code == 200, dossier_resp.text
    data = dossier_resp.json()
    assert data["proof_id"]
    assert data["status"] == "OK_OPTIMIZED"
    assert data["baseline"]["duty_rate"] == pytest.approx(37.5)
    assert data["optimized"]["suggestion"]["optimized_duty_rate"] == pytest.approx(3.0)


@pytest.mark.usefixtures("system_client")
def test_tariff_sku_dossier_v2_questions(system_client: TestClient):
    sku_id = "SKU-ELEC-AMBIG"
    sku_payload = {
        "sku_id": sku_id,
        "description": "Electronics enclosure with harness and PCB slot",
        "declared_value_per_unit": 120.0,
    }
    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text

    dossier_resp = system_client.post(f"/api/tariff/skus/{sku_id}/dossier", json={"optimize": True})
    assert dossier_resp.status_code == 200, dossier_resp.text
    body = dossier_resp.json()
    assert body["status"] == "OK_BASELINE_ONLY"
    assert body["questions"]["questions"]
    question = body["questions"]["questions"][0]
    assert question["why_needed"]
    assert question["accepted_evidence_types"]
