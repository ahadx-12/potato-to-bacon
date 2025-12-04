import importlib

import pytest
from fastapi.testclient import TestClient

from potatobacon.api.security import set_rate_limit


@pytest.fixture()
def app_with_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CALE_API_KEYS", "system-key")
    monkeypatch.setenv("CALE_RATE_LIMIT_PER_MINUTE", "30")
    set_rate_limit(30)

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


@pytest.fixture()
def bulk_manifest(authed_client):
    payload = {
        "domain": "tax",
        "sources": [
            {
                "id": "US_TAX",
                "text": "US corporations must pay corporate income tax on domestic profits.",
                "jurisdiction": "US",
                "statute": "IRC",
                "section": "11",
                "enactment_year": 2017,
            },
            {
                "id": "IE_TAX",
                "text": "Irish entities may claim R&D credits reducing effective tax.",
                "jurisdiction": "IE",
                "statute": "TCA",
                "section": "766",
                "enactment_year": 2020,
            },
            {
                "id": "KY_TAX",
                "text": "Kentucky allows deductions for manufacturing investments.",
                "jurisdiction": "KY",
                "statute": "KYREV",
                "section": "141",
                "enactment_year": 2018,
            },
        ],
        "options": {"replace_existing": True},
    }
    response = authed_client.post("/v1/manifest/bulk_ingest", json=payload)
    assert response.status_code == 200, response.text
    manifest_hash = response.json()["manifest_hash"]
    assert manifest_hash
    return manifest_hash
