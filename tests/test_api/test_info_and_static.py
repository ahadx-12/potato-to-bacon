from fastapi.testclient import TestClient

from potatobacon.api.app import app


client = TestClient(app)


def test_info_endpoint_exposes_metadata() -> None:
    response = client.get("/v1/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "0.1.0"
    assert "dimensional_fast" in payload["validators"]
    assert any("derivatives" in item for item in payload["dsl_features"])


def test_static_mounts_serve_docs_and_examples() -> None:
    docs_response = client.get("/static/docs/quickstart.md")
    assert docs_response.status_code == 200
    assert "Quickstart" in docs_response.text

    example_response = client.get("/static/examples/01_ke.dsl")
    assert example_response.status_code == 200
    assert "kinetic_energy" in example_response.text
