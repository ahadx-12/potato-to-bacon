import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_tariff_sku_dossier_electronics_baseline(system_client: TestClient):
    payload = {
        "description": "Electronics enclosure with PCB and wiring harness",
        "declared_value_per_unit": 120.0,
        "annual_volume": 1000,
        "include_fact_evidence": True,
    }

    resp = system_client.post("/api/tariff/sku/dossier", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] in {"OK_OPTIMIZED", "OK_BASELINE_ONLY"}
    assert data["baseline_candidates"] or data["compiled_facts"]
    assert data["why_not_optimized"] is not None


@pytest.mark.usefixtures("system_client")
def test_tariff_sku_dossier_apparel_baseline(system_client: TestClient):
    payload = {
        "description": "Woven cotton jacket with coating",
        "declared_value_per_unit": 80.0,
        "annual_volume": 500,
    }

    resp = system_client.post("/api/tariff/sku/dossier", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] in {"OK_OPTIMIZED", "OK_BASELINE_ONLY"}
    assert data["baseline_candidates"] or data["compiled_facts"]


@pytest.mark.usefixtures("system_client")
def test_tariff_sku_dossier_fastener_optimized(system_client: TestClient):
    payload = {
        "description": "Chassis bolt fastener for EV frame",
        "declared_value_per_unit": 200.0,
        "annual_volume": 10000,
    }

    resp = system_client.post("/api/tariff/sku/dossier", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "OK_OPTIMIZED"
    assert data["best_optimization"] is not None
    assert data["baseline_candidates"]
