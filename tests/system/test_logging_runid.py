from __future__ import annotations

import importlib
import logging

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("CALE_STORAGE_PATH", str(tmp_path / "persistence.db"))
    monkeypatch.setenv("CALE_API_KEYS", "dev-key")

    def _factory():
        import potatobacon.api.app as api_app

        importlib.reload(api_app)
        from potatobacon.api.security import set_rate_limit

        set_rate_limit(30)
        client = TestClient(api_app.app, headers={"X-API-Key": "dev-key"})
        return api_app, client

    return _factory


def _hunt_payload() -> dict:
    return {
        "jurisdictions": ["US"],
        "domain": "tax",
        "objective": "MAXIMIZE(net_after_tax_income)",
        "constraints": {"entity_type": "llc"},
        "risk_tolerance": "medium",
    }


def test_runid_and_redaction(client_factory, caplog):
    api_app, client = client_factory()
    caplog.set_level(logging.INFO, logger="potatobacon")
    try:
        resp = client.post("/api/law/arbitrage/hunt", json=_hunt_payload())
        assert resp.status_code == 200
    finally:
        client.close()

    run_ids = set()
    api_keys = set()
    for record in caplog.records:
        payload = getattr(record, "payload", {})
        if not payload:
            continue
        if payload.get("run_id"):
            run_ids.add(payload["run_id"])
        if "api_key" in payload:
            api_keys.add(payload["api_key"])

    assert len(run_ids) == 1
    assert all(val != "dev-key" for val in api_keys)
