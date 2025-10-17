from pathlib import Path

from fastapi.testclient import TestClient

from potatobacon.api.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_translate_validate_codegen_manifest() -> None:
    dsl = Path("examples/01_ke.dsl").read_text()

    translate = client.post("/v1/translate", json={"dsl": dsl})
    assert translate.status_code == 200
    assert translate.json()["success"] is True

    validate = client.post(
        "/v1/validate",
        json={
            "dsl": dsl,
            "domain": "classical",
            "units": {"m": "kg", "v": "m/s"},
            "result_unit": "J",
        },
    )
    assert validate.status_code == 200
    assert validate.json()["ok"] is True

    codegen = client.post("/v1/codegen", json={"dsl": dsl, "name": "ke"})
    assert codegen.status_code == 200
    assert "def ke" in codegen.json()["code"]

    manifest = client.post(
        "/v1/manifest",
        json={
            "dsl": dsl,
            "domain": "classical",
            "units": {"m": "kg", "v": "m/s"},
            "result_unit": "J",
        },
    )
    assert manifest.status_code == 200
    assert "manifest_hash" in manifest.json()
