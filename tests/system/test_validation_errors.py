"""System-level validation error shaping tests."""

from fastapi.testclient import TestClient


def test_empty_jurisdictions_returns_structured_error(system_client: TestClient):
    response = system_client.post("/api/law/arbitrage/hunt", json={"jurisdictions": []})

    assert response.status_code == 422
    assert response.json() == {
        "error": "VALIDATION_ERROR",
        "fields": [
            {"path": "request.jurisdictions", "message": "At least one jurisdiction is required"}
        ],
    }


def test_unknown_objective_rejected_with_enum_hint(system_client: TestClient):
    response = system_client.post(
        "/api/law/arbitrage/hunt",
        json={"jurisdictions": ["US"], "objective": "INVALID(goal)"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    field = body["fields"][0]
    assert field["path"] == "request.objective"
    assert "MAXIMIZE(net_after_tax_income)" in field["message"]


def test_malformed_types_are_normalized(system_client: TestClient):
    response = system_client.post(
        "/api/law/arbitrage/hunt",
        json={"jurisdictions": "US", "objective": 123, "constraints": "nope"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "VALIDATION_ERROR"
    messages = {field["path"]: field["message"] for field in body["fields"]}
    assert "request.jurisdictions" in messages
    assert "request.objective" in messages
    assert "request.constraints" in messages
    assert "system-key" not in response.text
