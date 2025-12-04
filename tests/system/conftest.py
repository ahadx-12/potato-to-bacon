"""Shared fixtures for system-level CALE-LAW tests."""

import importlib
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
