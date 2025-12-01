from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from potatobacon.api.app import app
from potatobacon.api.security import set_rate_limit


@pytest.fixture()
def authed_client(monkeypatch):
    monkeypatch.setenv("CALE_API_KEYS", "test-key")
    monkeypatch.setenv("CALE_RATE_LIMIT_PER_MINUTE", "5")
    set_rate_limit(5)
    with TestClient(app, headers={"X-API-Key": "test-key"}) as client:
        yield client


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


def test_missing_api_key_rejected(monkeypatch):
    monkeypatch.setenv("CALE_API_KEYS", "secret")
    with TestClient(app) as client:
        resp = client.post("/v1/law/analyze", json=_law_request_payload())
    assert resp.status_code == 401
    assert resp.json()["detail"]["message"] == "Missing API key"


def test_rate_limit_triggers(monkeypatch):
    monkeypatch.setenv("CALE_API_KEYS", "limit-key")
    set_rate_limit(2)
    with TestClient(app, headers={"X-API-Key": "limit-key"}) as client:
        for _ in range(2):
            ok = client.post("/v1/law/analyze", json=_law_request_payload())
            assert ok.status_code == 200
        limited = client.post("/v1/law/analyze", json=_law_request_payload())
        assert limited.status_code == 429
    set_rate_limit(5)


def test_bulk_manifest_ingest_updates_active_manifest(authed_client):
    payload = {
        "domain": "tax",
        "sources": [
            {"id": "US_IRC_61", "text": "Section 61 income includes all income."},
            {"id": "US_IRC_162", "text": "Section 162 allows deductions for expenses."},
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


def test_arbitrage_job_completion(authed_client):
    req = {
        "jurisdictions": ["Canada.Federal"],
        "domain": "tax",
        "objective": "MAXIMIZE(net_after_tax_income)",
        "constraints": {"consent": True},
        "risk_tolerance": "low",
    }
    submitted = authed_client.post("/api/law/arbitrage/hunt/job", json=req)
    assert submitted.status_code == 200
    job_id = submitted.json()["job_id"]

    for _ in range(10):
        status = authed_client.get(f"/api/law/jobs/{job_id}")
        assert status.status_code == 200
        payload = status.json()
        if payload["status"] == "completed":
            assert payload["result"]
            break
        time.sleep(0.1)
    else:
        pytest.fail("Job did not complete in time")


def test_pdf_ingestion_updates_manifest(authed_client, tmp_path):
    pdf_path = "tests/fixtures/test_statute.pdf"
    with open(pdf_path, "rb") as handle:
        files = {"file": ("test_statute.pdf", handle, "application/pdf")}
        response = authed_client.post(
            "/v1/manifest/ingest_pdf",
            data={"domain": "securities", "base_id": "SEC"},
            files=files,
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["manifest_hash"]
    assert body["extracted_sections"] >= 1

    analyze = authed_client.post("/v1/law/analyze", json=_law_request_payload())
    assert analyze.status_code == 200
    assert analyze.json().get("manifest_hash") == body["manifest_hash"]
