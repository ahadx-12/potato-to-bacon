import pytest


@pytest.mark.usefixtures("system_client")
def test_tariff_parse_endpoint(system_client):
    payload = {
        "description": "Converse-like canvas sneaker with rubber sole",
        "bom_text": "Rubber sole with textile upper",
    }

    response = system_client.post("/api/tariff/parse", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["product_spec"]["product_category"] == "footwear"
    assert data["compiled_facts"]
    assert data["fact_evidence"]
    assert data["extraction_evidence"]
