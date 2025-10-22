from __future__ import annotations

from fastapi.testclient import TestClient

from potatobacon.api.app import app


def test_cale_static_demo_served(monkeypatch) -> None:
    monkeypatch.delenv("CALE_DISABLE_STARTUP_INIT", raising=False)
    with TestClient(app) as client:
        response = client.get("/demo/cale/")
        assert response.status_code == 200
        assert "<title>CALE Demo</title>" in response.text
