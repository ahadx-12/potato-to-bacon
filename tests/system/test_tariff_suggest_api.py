import pytest


@pytest.mark.usefixtures("system_client")
def test_converse_suggest(system_client):
    payload = {
        "sku_id": "CONVERSE_DEMO",
        "description": "Canvas sneaker with rubber sole",
        "declared_value_per_unit": 100.0,
        "annual_volume": 10_000,
        "top_k": 3,
    }

    response = system_client.post("/api/tariff/suggest", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["status"] == "OK"
    assert data["generated_candidates_count"] >= 1

    suggestion = data["suggestions"][0]
    assert pytest.approx(suggestion["baseline_duty_rate"], rel=1e-6) == 37.5
    assert pytest.approx(suggestion["optimized_duty_rate"], rel=1e-6) == 3.0
    assert pytest.approx(suggestion["annual_savings_value"], rel=1e-6) == 345_000.0
    assert isinstance(suggestion.get("proof_id"), str) and suggestion.get("proof_id")


@pytest.mark.usefixtures("system_client")
def test_tesla_bolt_suggest(system_client):
    payload = {
        "sku_id": "TESLA_BOLT_X1",
        "description": "Chassis bolt fastener for EV frame",
        "declared_value_per_unit": 200.0,
        "annual_volume": 50_000,
        "top_k": 3,
    }

    response = system_client.post("/api/tariff/suggest", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["status"] == "OK"

    suggestion = data["suggestions"][0]
    assert pytest.approx(suggestion["baseline_duty_rate"], rel=1e-6) == 6.5
    assert pytest.approx(suggestion["optimized_duty_rate"], rel=1e-6) == 2.5
    assert pytest.approx(suggestion["annual_savings_value"], rel=1e-6) == 400_000.0
    assert isinstance(suggestion.get("proof_id"), str) and suggestion.get("proof_id")


@pytest.mark.usefixtures("system_client")
def test_unknown_suggest(system_client):
    payload = {
        "description": "random gadget with minimal info",
        "declared_value_per_unit": 50.0,
    }

    response = system_client.post("/api/tariff/suggest", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["status"] == "NO_CANDIDATES"
    assert data["suggestions"] == []
