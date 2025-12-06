"""Shared fixtures for system-level CALE-LAW tests."""

import importlib
from contextlib import contextmanager
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from potatobacon.api.security import set_rate_limit
from potatobacon.law.jobs import job_manager


@pytest.fixture()
def system_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CALE_API_KEYS", "system-key")
    monkeypatch.setenv("CALE_RATE_LIMIT_PER_MINUTE", "100")
    set_rate_limit(100)
    job_manager.reset()

    import potatobacon.storage as storage_mod
    import potatobacon.manifest.store as store_mod
    import potatobacon.api.app as app_mod

    storage_mod = importlib.reload(storage_mod)
    store_mod = importlib.reload(store_mod)
    app_mod = importlib.reload(app_mod)

    return app_mod.app


@pytest.fixture()
def system_client(system_app) -> Iterator[TestClient]:
    with TestClient(system_app, headers={"X-API-Key": "system-key"}) as client:
        yield client


@pytest.fixture()
def authed_client(system_client) -> Iterator[TestClient]:
    """Alias to keep system tests consistent with law fixtures."""

    yield system_client


@pytest.fixture()
def bulk_manifest(system_client):
    """Seed a minimal manifest for arbitrage tests."""

    payload = {
        "domain": "tax",
        "sources": [
            {"id": "TEST_RATE_1", "text": "Trading income taxed at standard rate."},
            {"id": "TEST_RATE_2", "text": "Passive income taxed at higher rate."},
        ],
        "options": {"replace_existing": True},
    }

    response = system_client.post("/v1/manifest/bulk_ingest", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["manifest_hash"]


@pytest.fixture()
def make_client(monkeypatch, tmp_path):
    """Factory to produce isolated API clients with configurable env settings."""

    @contextmanager
    def _factory(headers: dict | None = None, extra_env: dict | None = None):
        env = {
            "PTB_DATA_ROOT": str(tmp_path),
            "CALE_API_KEYS": "dev-key,alt-key",
            "CALE_RATE_LIMIT_PER_MINUTE": "100",
        }
        if extra_env:
            env.update(extra_env)
        for key, val in env.items():
            monkeypatch.setenv(key, str(val))

        import potatobacon.storage as storage_mod
        import potatobacon.manifest.store as store_mod
        import potatobacon.api.app as app_mod

        storage_mod = importlib.reload(storage_mod)
        store_mod = importlib.reload(store_mod)
        app_mod = importlib.reload(app_mod)

        from potatobacon.api.security import set_rate_limit

        set_rate_limit(int(env.get("CALE_RATE_LIMIT_PER_MINUTE", "100")))

        default_headers = {}
        if headers:
            default_headers.update(headers)

        client = TestClient(app_mod.app, headers=default_headers)
        try:
            yield client, env
        finally:
            client.close()

    return _factory
