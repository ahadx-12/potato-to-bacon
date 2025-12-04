from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from potatobacon.sdk.client import PBClient, PBConfig


@pytest.fixture()
def client_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("CALE_STORAGE_PATH", str(tmp_path / "persistence.db"))
    monkeypatch.setenv("CALE_API_KEYS", "dev-key")

    def _factory():
        import potatobacon.api.app as api_app

        importlib.reload(api_app)
        from potatobacon.api.security import set_rate_limit

        set_rate_limit(50)
        client = TestClient(api_app.app, headers={"X-API-Key": "dev-key"})
        return api_app, client

    return _factory


def _rules() -> tuple[dict, dict]:
    rule1 = {
        "text": "Organizations MUST collect personal data IF consent.",
        "jurisdiction": "Canada.Federal",
        "statute": "PIPEDA",
        "section": "7(3)",
        "enactment_year": 2000,
    }
    rule2 = {
        "text": "Security agencies MUST NOT collect personal data IF emergency.",
        "jurisdiction": "Canada.Federal",
        "statute": "Anti-Terrorism Act",
        "section": "83.28",
        "enactment_year": 2001,
    }
    return rule1, rule2


def test_sdk_roundtrip_calls(client_factory):
    api_app, client = client_factory()
    try:
        sdk = PBClient(cfg=PBConfig(base_url=str(client.base_url)), session=client)
        assert sdk._session.headers["X-API-Key"] == "dev-key"

        version = sdk.version()
        assert version.engine_version

        sources = [
            {
                "id": "sample",
                "text": Path("tests/fixtures/sample_text.txt").read_text(),
                "jurisdiction": "US",
                "statute": "Demo",
                "section": "1",
                "enactment_year": 2020,
            }
        ]
        ingest = sdk.bulk_ingest("tax", sources)
        assert ingest["manifest_hash"]

        rule1, rule2 = _rules()
        analyze = sdk.analyze(rule1, rule2)
        assert analyze["scenario_metrics"]

        hunt_req = {
            "jurisdictions": ["US"],
            "domain": "tax",
            "objective": "MAXIMIZE(net_after_tax_income)",
            "constraints": {"entity_type": "llc"},
            "risk_tolerance": "medium",
        }
        hunt = sdk.hunt(hunt_req)
        assert hunt["metrics"]
        asset_id = hunt["id"]

        assets = sdk.assets.list(jurisdiction="US", from_date="2025-01-01", limit=1)
        assert assets["items"]
        detail = sdk.assets.get(asset_id)
        assert detail.provenance_chain
        assert detail.dependency_graph is not None
    finally:
        client.close()
