import pytest
from fastapi.testclient import TestClient


def _assert_suggestion_fields(suggestion: dict):
    assert suggestion["lever_id"]
    assert suggestion["proof_id"]
    assert suggestion["proof_payload_hash"]
    assert suggestion["tariff_manifest_hash"]


@pytest.mark.usefixtures("system_client")
def test_electronics_dossier_produces_suggestions(system_client: TestClient):
    payload = {
        "description": "Electronics enclosure with PCB and wiring harness for automotive gateway",
        "declared_value_per_unit": 150.0,
        "annual_volume": 2000,
        "include_fact_evidence": True,
        "seed": 2025,
    }

    first = system_client.post("/api/tariff/suggest", json=payload)
    second = system_client.post("/api/tariff/suggest", json=payload)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    first_body = first.json()
    second_body = second.json()

    assert first_body["status"] in {"OK_OPTIMIZED", "OK_BASELINE_ONLY"}
    assert first_body["suggestions"], "electronics should yield at least one lever suggestion"
    _assert_suggestion_fields(first_body["suggestions"][0])

    assert first_body["suggestions"][0]["lever_id"] == second_body["suggestions"][0]["lever_id"]
    assert first_body["suggestions"][0]["proof_payload_hash"] == second_body["suggestions"][0]["proof_payload_hash"]


@pytest.mark.usefixtures("system_client")
def test_apparel_dossier_has_blend_lever(system_client: TestClient):
    payload = {
        "description": "Polyester cotton blend woven jacket with coating for outdoor use",
        "declared_value_per_unit": 90.0,
        "annual_volume": 500,
        "seed": 2025,
    }

    response = system_client.post("/api/tariff/suggest", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["suggestions"], "apparel blend should surface lever suggestions"
    first_suggestion = body["suggestions"][0]
    _assert_suggestion_fields(first_suggestion)
