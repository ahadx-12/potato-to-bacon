import importlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient


def test_end_to_end_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CALE_API_KEYS", "dev-key")

    import potatobacon.storage as storage_mod
    import potatobacon.manifest.store as store_mod
    import potatobacon.api.app as app_mod

    storage_mod = importlib.reload(storage_mod)
    store_mod = importlib.reload(store_mod)
    app_mod = importlib.reload(app_mod)

    client = TestClient(app_mod.app, headers={"X-API-Key": "dev-key"})

    dsl_text = Path("examples/01_ke.dsl").read_text()

    translate_resp = client.post("/v1/translate", json={"dsl": dsl_text})
    assert translate_resp.status_code == 200
    translate_payload = translate_resp.json()
    assert translate_payload["success"] is True

    validate_resp = client.post(
        "/v1/validate",
        json={
            "dsl": dsl_text,
            "domain": "classical",
            "units": {"m": "kg", "v": "m/s"},
            "result_unit": "J",
        },
    )
    assert validate_resp.status_code == 200
    assert validate_resp.json()["ok"] is True

    codegen_resp = client.post("/v1/codegen", json={"dsl": dsl_text, "name": "ke"})
    assert codegen_resp.status_code == 200
    code = codegen_resp.json()["code"]
    namespace: Dict[str, Any] = {}
    exec(code, namespace)
    result = namespace["ke"](m=2.0, v=3.0)
    assert result == 9.0

    manifest_resp = client.post(
        "/v1/manifest",
        json={
            "dsl": dsl_text,
            "domain": "classical",
            "units": {"m": "kg", "v": "m/s"},
            "result_unit": "J",
        },
    )
    assert manifest_resp.status_code == 200
    manifest_payload = manifest_resp.json()

    manifest_get = client.get(f"/v1/manifest/{manifest_payload['manifest_hash']}")
    assert manifest_get.status_code == 200
    assert "canonical" in manifest_get.json()

    info_resp = client.get("/v1/info")
    assert info_resp.status_code == 200
    assert "pde_class" in info_resp.json()["validators"]

    docker_path = shutil.which("docker")
    if docker_path:
        repo_root = Path(__file__).resolve().parents[2]
        build = subprocess.run([docker_path, "build", "-q", "."], cwd=repo_root, check=False)
        assert build.returncode == 0
