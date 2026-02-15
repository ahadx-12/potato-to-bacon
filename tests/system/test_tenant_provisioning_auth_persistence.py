from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
PROVISION_SCRIPT = REPO_ROOT / "scripts" / "provision_tenant.py"


def test_provisioned_key_authenticates_across_fresh_app_context(monkeypatch, tmp_path):
    tenant_id = "tenant_persisted"
    api_key = "ptb_persisted_test_key"

    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PTB_STORAGE_BACKEND", "jsonl")
    monkeypatch.delenv("CALE_API_KEYS", raising=False)

    env = {
        **dict(os.environ),
        "PTB_DATA_ROOT": str(tmp_path),
        "PTB_STORAGE_BACKEND": "jsonl",
    }

    result = subprocess.run(
        [
            sys.executable,
            str(PROVISION_SCRIPT),
            "--name",
            "Persisted Tenant",
            "--tenant-id",
            tenant_id,
            "--api-key",
            api_key,
            "--plan",
            "professional",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    import potatobacon.api.tenants as tenants_mod
    import potatobacon.api.security as security_mod
    import potatobacon.api.app as app_mod

    tenants_mod = importlib.reload(tenants_mod)
    security_mod = importlib.reload(security_mod)
    app_mod = importlib.reload(app_mod)

    with TestClient(app_mod.app, headers={"X-API-Key": api_key}) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200

        csv_body = "part_id,description,qty,unit_cost,country_of_origin\nP-1,Widget,1,10.0,US\n"
        upload = client.post(
            "/v1/bom/upload",
            files={"file": ("bom.csv", csv_body.encode("utf-8"), "text/csv")},
        )
        assert upload.status_code == 200, upload.text
        payload = upload.json()
        assert payload["status"] == "parsed"
        assert payload["parseable_rows"] >= 1
