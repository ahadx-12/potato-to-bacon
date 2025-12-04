from __future__ import annotations

import importlib
from typing import Callable, Tuple

import pytest
from fastapi.testclient import TestClient


def _hunt_request() -> dict:
    return {
        "jurisdictions": ["US"],
        "domain": "tax",
        "objective": "MAXIMIZE(net_after_tax_income)",
        "constraints": {"entity_type": "llc"},
        "risk_tolerance": "medium",
    }


@pytest.fixture()
def client_factory(tmp_path, monkeypatch) -> Callable[[], Tuple[object, TestClient]]:
    monkeypatch.setenv("CALE_STORAGE_PATH", str(tmp_path / "persistence.db"))
    monkeypatch.setenv("CALE_API_KEYS", "dev-key")

    def _factory() -> Tuple[object, TestClient]:
        import potatobacon.api.app as api_app

        importlib.reload(api_app)
        from potatobacon.api.security import set_rate_limit

        set_rate_limit(20)
        client = TestClient(api_app.app, headers={"X-API-Key": "dev-key"})
        return api_app, client

    return _factory


def test_assets_survive_restart(client_factory):
    api_app, client = client_factory()
    try:
        first = client.post("/api/law/arbitrage/hunt", json=_hunt_request())
        second = client.post("/api/law/arbitrage/hunt", json=_hunt_request())
        assert first.status_code == 200 and second.status_code == 200
        list_resp = client.get(
            "/api/law/arbitrage/assets",
            params={"jurisdiction": "US", "from": "2025-01-01", "limit": 1},
        )
        assert list_resp.status_code == 200
        items = list_resp.json()["items"]
        assert items and "metrics" in items[0]
        if len(items) == 1:
            assert list_resp.json().get("next_cursor")
        detail = client.get(f"/api/law/arbitrage/assets/{items[0]['id']}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["provenance_chain"]
        assert body["dependency_graph"] is not None
        asset_id = items[0]["id"]
    finally:
        client.close()

    api_app, client_restart = client_factory()
    try:
        list_again = client_restart.get(
            "/api/law/arbitrage/assets",
            params={"jurisdiction": "US", "from": "2025-01-01", "limit": 1},
        )
        assert list_again.status_code == 200
        items_again = list_again.json()["items"]
        assert items_again
        detail_again = client_restart.get(f"/api/law/arbitrage/assets/{asset_id}")
        assert detail_again.status_code == 200
        assert detail_again.json()["id"] == asset_id
    finally:
        client_restart.close()
