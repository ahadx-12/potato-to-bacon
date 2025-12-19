import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_tariff_suggest_determinism_payload_hash(system_client: TestClient):
    request = {
        "description": "Canvas sneaker with rubber sole",
        "declared_value_per_unit": 150.0,
        "annual_volume": 10000,
        "include_fact_evidence": True,
        "seed": 4242,
        "top_k": 2,
    }

    first = system_client.post("/api/tariff/suggest", json=request)
    second = system_client.post("/api/tariff/suggest", json=request)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    first_payload = first.json()
    second_payload = second.json()

    assert first_payload["status"] == "OK_OPTIMIZED"
    assert second_payload["status"] == "OK_OPTIMIZED"

    first_best = first_payload["suggestions"][0]
    second_best = second_payload["suggestions"][0]

    assert first_best["best_mutation"] == second_best["best_mutation"]
    assert first_best["baseline_duty_rate"] == second_best["baseline_duty_rate"]
    assert first_best["optimized_duty_rate"] == second_best["optimized_duty_rate"]
    assert first_best["tariff_manifest_hash"] == second_best["tariff_manifest_hash"]
    assert first_best["proof_payload_hash"] == second_best["proof_payload_hash"]
    assert first_best["active_codes_optimized"] == second_best["active_codes_optimized"]

    assert first_best["proof_id"]
    assert second_best["proof_id"]
