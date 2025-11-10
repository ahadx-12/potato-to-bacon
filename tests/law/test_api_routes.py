from __future__ import annotations

from fastapi.testclient import TestClient

from potatobacon.api.app import app


client = TestClient(app)


def test_summary_endpoint_returns_json() -> None:
    response = client.get("/api/law/tax/summary")
    assert response.status_code == 200
    payload = response.json()
    assert {"sections_total", "pairs_total", "top_sections"}.issubset(payload.keys())


def test_sections_pagination_defaults() -> None:
    response = client.get("/api/law/tax/sections")
    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 1
    assert payload["page_size"] == 50


def test_pairs_limit() -> None:
    response = client.get("/api/law/tax/pairs?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] <= 5
