import importlib
import time
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from potatobacon.api.security import set_rate_limit


@pytest.fixture()
def app_with_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CALE_API_KEYS", "system-key")
    monkeypatch.setenv("CALE_RATE_LIMIT_PER_MINUTE", "10")
    set_rate_limit(10)

    import potatobacon.storage as storage_mod
    import potatobacon.manifest.store as store_mod
    import potatobacon.api.app as app_mod

    storage_mod = importlib.reload(storage_mod)
    store_mod = importlib.reload(store_mod)
    app_mod = importlib.reload(app_mod)

    return app_mod.app


@pytest.fixture()
def authed_client(app_with_auth):
    with TestClient(app_with_auth, headers={"X-API-Key": "system-key"}) as client:
        yield client


def _law_request_payload() -> Dict[str, Dict[str, object]]:
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


def test_version_and_rate_limiting(app_with_auth):
    with TestClient(app_with_auth, headers={"X-API-Key": "system-key"}) as client:
        version = client.get("/v1/version")
        assert version.status_code == 200
        body = version.json()
        assert body["engine_version"]
        assert "manifest_hash" in body

        unauth = client.post("/v1/law/analyze", json=_law_request_payload())
        assert unauth.status_code == 200

    # Missing and invalid API keys should be rejected
    with TestClient(app_with_auth) as open_client:
        resp = open_client.post("/v1/law/analyze", json=_law_request_payload())
        assert resp.status_code == 401
        assert resp.json()["detail"]["message"] == "Missing API key"

    with TestClient(app_with_auth, headers={"X-API-Key": "bad-key"}) as bad_client:
        resp = bad_client.post("/v1/law/analyze", json=_law_request_payload())
        assert resp.status_code == 401
        assert resp.json()["detail"]["message"] == "Invalid API key"

    # Rate limiting kicks in after the configured quota
    set_rate_limit(2)
    with TestClient(app_with_auth, headers={"X-API-Key": "system-key"}) as limited_client:
        first = limited_client.post("/v1/law/analyze", json=_law_request_payload())
        second = limited_client.post("/v1/law/analyze", json=_law_request_payload())
        assert first.status_code == 200
        assert second.status_code == 200
        limited = limited_client.post("/v1/law/analyze", json=_law_request_payload())
        assert limited.status_code == 429
    set_rate_limit(10)


def test_bulk_ingest_and_analyze_roundtrip(authed_client):
    payload = {
        "domain": "tax",
        "sources": [
            {"id": "INGEST_1", "text": "Income includes wages and investment gains."},
            {"id": "INGEST_2", "text": "Deductions are allowed for necessary business expenses."},
        ],
        "options": {"replace_existing": True},
    }
    response = authed_client.post("/v1/manifest/bulk_ingest", json=payload)
    assert response.status_code == 200, response.text
    manifest_hash = response.json()["manifest_hash"]
    assert manifest_hash

    analyze = authed_client.post("/v1/law/analyze", json=_law_request_payload())
    assert analyze.status_code == 200
    assert analyze.json().get("manifest_hash") == manifest_hash


@pytest.mark.parametrize(
    "base_id",
    ["TEST_SEC", None],
)
def test_pdf_ingest_and_analysis(authed_client, base_id):
    pdf_path = Path("tests/fixtures/test_statute.pdf")
    with pdf_path.open("rb") as handle:
        files = {"file": (pdf_path.name, handle, "application/pdf")}
        response = authed_client.post(
            "/v1/manifest/ingest_pdf",
            data={"domain": "securities", "base_id": base_id or ""},
            files=files,
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["manifest_hash"]
    assert body["extracted_sections"] >= 1

    analyze = authed_client.post("/v1/law/analyze", json=_law_request_payload())
    assert analyze.status_code == 200
    assert analyze.json().get("manifest_hash") == body["manifest_hash"]


def test_law_analyze_reports_scenarios(authed_client):
    response = authed_client.post("/v1/law/analyze", json=_law_request_payload())
    assert response.status_code == 200
    body = response.json()

    scenario_metrics = body.get("scenario_metrics", {})
    assert "summary" in scenario_metrics
    assert scenario_metrics.get("samples")
    first_sample = scenario_metrics["samples"][0]
    for field in ["entropy", "contradiction", "kappa", "value_estimate", "risk", "score"]:
        assert field in first_sample


def test_arbitrage_sync_single_and_multi(authed_client):
    single_req = {
        "jurisdictions": ["US"],
        "domain": "tax",
        "objective": "MAXIMIZE(net_after_tax_income)",
        "constraints": {"consent": True},
        "risk_tolerance": "low",
    }
    multi_req = {**single_req, "jurisdictions": ["US", "Ireland", "Cayman"]}

    single = authed_client.post("/api/law/arbitrage/hunt", json=single_req)
    multi = authed_client.post("/api/law/arbitrage/hunt", json=multi_req)

    for resp in (single, multi):
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["golden_scenario"]
        metrics = payload["metrics"]
        for key in ["value", "entropy", "kappa", "risk", "score"]:
            assert key in metrics
        assert payload["proof_trace"]
        assert isinstance(payload.get("risk_flags", []), list)


def test_arbitrage_job_flow(authed_client):
    req = {
        "jurisdictions": ["Canada.Federal"],
        "domain": "tax",
        "objective": "MAXIMIZE(net_after_tax_income)",
        "constraints": {"consent": True},
        "risk_tolerance": "medium",
    }
    submitted = authed_client.post("/api/law/arbitrage/hunt/job", json=req)
    assert submitted.status_code == 200
    job_id = submitted.json()["job_id"]

    final_state: Dict[str, object] | None = None
    for _ in range(15):
        status = authed_client.get(f"/api/law/jobs/{job_id}")
        assert status.status_code == 200, status.text
        payload = status.json()
        if payload["status"] == "completed":
            final_state = payload
            break
        time.sleep(0.1)
    else:
        pytest.fail("Job did not complete in time")

    assert final_state is not None
    assert final_state["result"]
    metrics = final_state["result"]["metrics"]
    for key in ["value", "entropy", "kappa", "risk", "score"]:
        assert key in metrics
    assert final_state["engine_version"]


def test_error_handling_responses(app_with_auth):
    with TestClient(app_with_auth, headers={"X-API-Key": "system-key"}) as client:
        bad_analyze = client.post("/v1/law/analyze", json={"rule1": {"text": ""}})
        assert bad_analyze.status_code == 422

        missing_job = client.get("/api/law/jobs/does-not-exist")
        assert missing_job.status_code == 404
        assert missing_job.json()["detail"] == "Job not found"

        malformed_arbitrage = client.post(
            "/api/law/arbitrage/hunt",
            json={"jurisdictions": "US", "domain": None, "objective": 123},
        )
        assert malformed_arbitrage.status_code == 422
