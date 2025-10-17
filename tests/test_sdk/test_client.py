import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def test_sdk_client_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))

    import potatobacon.storage as storage_mod
    import potatobacon.manifest.store as store_mod
    import potatobacon.api.app as app_mod
    from potatobacon.sdk.client import PBClient, PBConfig

    storage_mod = importlib.reload(storage_mod)
    store_mod = importlib.reload(store_mod)
    app_mod = importlib.reload(app_mod)

    client = PBClient(cfg=PBConfig(base_url="http://testserver"), session=TestClient(app_mod.app))

    dsl_text = Path("examples/01_ke.dsl").read_text()

    translate = client.translate(dsl_text)
    assert translate.success is True

    validate = client.validate(
        dsl_text,
        units={"m": "kg", "v": "m/s"},
        result_unit="J",
    )
    assert validate.ok is True

    codegen = client.codegen(dsl_text, name="ke")
    assert "def ke" in codegen.code

    manifest = client.manifest(
        dsl_text,
        units={"m": "kg", "v": "m/s"},
        result_unit="J",
    )
    manifest_data = client.get_manifest(manifest.manifest_hash)
    assert manifest_data["code_digest"] == manifest.code_digest

    info = client.info()
    assert "pde_class" in info.validators
