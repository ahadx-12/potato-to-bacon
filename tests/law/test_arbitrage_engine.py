import os

from fastapi.testclient import TestClient

from potatobacon.api.app import app


os.environ["CALE_API_KEYS"] = "test-key"
client = TestClient(app, headers={"X-API-Key": "test-key"})


def _law_request_payload():
    return {
        "rule1": {
            "text": "Organizations MUST collect personal data IF consent.",
            "jurisdiction": "Canada.Federal",
            "statute": "PIPEDA",
            "section": "7(3)",
            "enactment_year": 2000,
        },
        "rule2": {
            "text": "Security agencies MUST NOT collect personal data IF emergency.",
            "jurisdiction": "Canada.Federal",
            "statute": "Anti-Terrorism Act",
            "section": "83.28",
            "enactment_year": 2001,
        },
    }


def test_analyze_includes_scenario_metrics():
    response = client.post("/v1/law/analyze", json=_law_request_payload())
    assert response.status_code == 200
    body = response.json()
    assert "scenario_metrics" in body
    metrics = body["scenario_metrics"]
    assert set(metrics.keys()) == {"summary", "samples"}
    assert metrics["summary"]["contradiction_probability"] >= 0.0
    assert isinstance(metrics["samples"], list)


def test_arbitrage_hunt_returns_dossier():
    request = {
        "jurisdictions": ["Canada.Federal"],
        "domain": "tax",
        "objective": "MAXIMIZE(net_after_tax_income)",
        "constraints": {"consent": True},
        "risk_tolerance": "medium",
    }
    response = client.post("/api/law/arbitrage/hunt", json=request)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) >= {"golden_scenario", "metrics", "proof_trace", "candidates"}
    assert payload["golden_scenario"]["jurisdictions"] == request["jurisdictions"]
    assert isinstance(payload["golden_scenario"].get("facts"), dict)

    metrics = payload["metrics"]
    assert "value_components" in metrics and metrics["value_components"]
    assert "risk_components" in metrics and metrics["risk_components"]

    provenance = payload.get("provenance_chain", [])
    assert provenance and provenance[0]["jurisdiction"]
    assert provenance[0]["rule_id"]

    graph = payload.get("dependency_graph", {}) or {}
    assert graph.get("nodes")

    assert isinstance(payload["candidates"], list)
    if payload["candidates"]:
        candidate = payload["candidates"][0]
        assert "scenario" in candidate and "metrics" in candidate
