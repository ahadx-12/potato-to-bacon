from __future__ import annotations

from fastapi.testclient import TestClient

from potatobacon.api.app import app


client = TestClient(app)


def test_dashboard_route_serves_html() -> None:
    response = client.get("/law/tax")
    assert response.status_code == 200
    assert "CALE-LAW" in response.text
    assert "tax-dashboard" in response.text
