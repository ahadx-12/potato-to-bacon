from __future__ import annotations

from fastapi.testclient import TestClient

from potatobacon.api.app import app
from potatobacon.cale.runtime import bootstrap


def _sample_payload() -> dict:
    rule = {
        "text": "Organizations MUST collect personal data IF consent.",
        "jurisdiction": "CA.Federal",
        "statute": "PIPEDA",
        "section": "7(3)",
        "enactment_year": 2000,
    }
    return {"rule1": rule, "rule2": rule}


def test_startup_initialises_services(monkeypatch) -> None:
    monkeypatch.delenv("CALE_DISABLE_STARTUP_INIT", raising=False)
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200
        assert getattr(client.app.state, "cale", None) is not None


def test_startup_can_be_skipped(monkeypatch) -> None:
    monkeypatch.setenv("CALE_DISABLE_STARTUP_INIT", "1")
    with TestClient(app) as client:
        resp = client.post("/v1/law/analyze", json=_sample_payload())
        assert resp.status_code == 503
        assert resp.json()["detail"] == "CALE services unavailable (not initialised)"
    monkeypatch.setenv("CALE_DISABLE_STARTUP_INIT", "0")
    services = bootstrap()
    if services is not None:
        app.state.cale = services
