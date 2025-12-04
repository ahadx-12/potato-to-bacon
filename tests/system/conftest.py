import contextlib
import importlib
import shutil
from pathlib import Path
from typing import Callable, ContextManager, Dict, Iterator, Tuple
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def make_client(monkeypatch) -> Callable[[Dict[str, str] | None, Dict[str, str] | None], ContextManager[Tuple[TestClient, Path]]]:
    """Factory for building isolated TestClient instances with fresh state."""

    @contextlib.contextmanager
    def _factory(
        extra_env: Dict[str, str] | None = None, headers: Dict[str, str] | None = None
    ) -> Iterator[Tuple[TestClient, Path]]:
        overrides = extra_env or {}
        data_root = Path("out/system-tests") / uuid4().hex
        if data_root.exists():
            shutil.rmtree(data_root)
        data_root.mkdir(parents=True, exist_ok=True)

        base_env = {
            "CALE_API_KEYS": "dev-key,alt-key",
            "PTB_DATA_ROOT": str(data_root),
            "CALE_RATE_LIMIT_PER_MINUTE": overrides.get("CALE_RATE_LIMIT_PER_MINUTE", "5"),
        }
        for key, value in base_env.items():
            monkeypatch.setenv(key, value)
        for key, value in overrides.items():
            monkeypatch.setenv(key, str(value))

        import potatobacon.storage as storage
        import potatobacon.api.security as security
        import potatobacon.api.app as app_module

        importlib.reload(storage)
        importlib.reload(security)
        importlib.reload(app_module)

        with TestClient(app_module.app, headers=headers) as client:
            yield client, data_root

    return _factory
