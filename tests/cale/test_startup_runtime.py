from __future__ import annotations

from fastapi.testclient import TestClient

import potatobacon.api.app as api_app_module
from potatobacon.api.app import app


def _sample_payload() -> dict:
    rule = {
        "text": "Organizations MUST collect personal data IF consent.",
        "jurisdiction": "CA.Federal",
        "statute": "PIPEDA",
        "section": "7(3)",
        "enactment_year": 2000,
    }
    return {"rule1": rule, "rule2": rule}


def test_startup_initialises_services() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert getattr(client.app.state, "cale", None) is not None


def test_startup_failure_returns_503(monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(api_app_module, "build_services", _boom)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["detail"] == "CALE not initialised"
