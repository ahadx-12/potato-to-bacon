from fastapi.testclient import TestClient

from potatobacon.api.app import app


client = TestClient(app)


def test_units_parse_endpoint_returns_warnings():
    response = client.post(
        "/v1/units/parse",
        json={"text": "m: kg\ninvalid"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["units"] == {"m": "kg"}
    assert data["warnings"]


def test_units_validate_endpoint_flags_errors():
    response = client.post(
        "/v1/units/validate",
        json={"units": {"good": "m", "bad": "m//s"}},
    )
    data = response.json()
    assert data["ok"] is False
    assert any(diag["symbol"] == "bad" for diag in data["diagnostics"])


def test_units_infer_endpoint_infers_energy():
    response = client.post(
        "/v1/units/infer",
        json={"dsl": "E = 0.5*m*v^2", "known": {"m": "kg", "v": "m/s"}},
    )
    data = response.json()
    assert data["ok"] is True
    assert data["units"]["E"] == "J"


def test_units_suggest_endpoint_uses_system():
    response = client.post(
        "/v1/units/suggest",
        json={"variables": ["m", "v"], "system": "SI"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["suggestions"]["m"] == "kg"
