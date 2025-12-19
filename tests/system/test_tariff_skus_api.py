import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_tariff_sku_registry_crud(system_client: TestClient):
    sku_payload = {
        "sku_id": "SKU-CONVERSE-DEMO",
        "description": "Canvas sneaker with rubber outsole",
        "origin_country": "VN",
        "declared_value_per_unit": 48.0,
        "annual_volume": 1200,
    }

    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text
    body = create_resp.json()
    assert body["status"] == "OK"
    assert body["sku_id"] == sku_payload["sku_id"]
    assert body["created"] is True

    fetch_resp = system_client.get(f"/api/tariff/skus/{sku_payload['sku_id']}")
    assert fetch_resp.status_code == 200, fetch_resp.text
    fetched = fetch_resp.json()
    assert fetched["sku_id"] == sku_payload["sku_id"]
    assert fetched["description"] == sku_payload["description"]

    list_resp = system_client.get("/api/tariff/skus")
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert any(item["sku_id"] == sku_payload["sku_id"] for item in listed)

    delete_resp = system_client.delete(f"/api/tariff/skus/{sku_payload['sku_id']}")
    assert delete_resp.status_code == 200, delete_resp.text
    assert delete_resp.json()["status"] == "DELETED"

    missing_resp = system_client.get(f"/api/tariff/skus/{sku_payload['sku_id']}")
    assert missing_resp.status_code == 404
