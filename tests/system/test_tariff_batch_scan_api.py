import pytest


@pytest.mark.usefixtures("system_client")
def test_tariff_batch_scan_api(system_client):
    payload = {
        "skus": [
            {
                "sku_id": "TESLA_BOLT_X1",
                "description": "Chassis bolt fastener for EV frame",
                "declared_value_per_unit": 200.0,
                "annual_volume": 50_000,
            },
            {
                "sku_id": "CONVERSE_DEMO",
                "description": "Canvas sneaker with rubber sole",
                "declared_value_per_unit": 100.0,
                "annual_volume": 10_000,
            },
            {
                "sku_id": "RANDOM_GADGET",
                "description": "random gadget",
                "declared_value_per_unit": 50.0,
            },
        ],
        "top_k_per_sku": 3,
        "max_results": 5,
        "seed": 2025,
    }

    response = system_client.post("/api/tariff/batch-scan", json=payload)

    assert response.status_code == 200, response.text

    data = response.json()
    assert data["status"] == "OK"
    assert data["total_skus"] == 3
    assert len(data["results"]) == 2

    assert [res["sku_id"] for res in data["results"]] == [
        "TESLA_BOLT_X1",
        "CONVERSE_DEMO",
    ]

    for res in data["results"]:
        assert res["best"]["proof_id"]

    skipped = {item["sku_id"]: item for item in data["skipped"]}
    assert skipped["RANDOM_GADGET"]["status"] == "NO_CANDIDATES"


@pytest.mark.usefixtures("system_client")
def test_tariff_batch_scan_risk_adjusted_rank(system_client):
    payload = {
        "skus": [
            {
                "sku_id": "TESLA_BOLT_X1",
                "description": "Chassis bolt fastener for EV frame",
                "declared_value_per_unit": 200.0,
                "annual_volume": 50_000,
            }
        ],
        "top_k_per_sku": 2,
        "risk_adjusted_ranking": True,
        "risk_penalty": 0.5,
        "seed": 2025,
    }

    response = system_client.post("/api/tariff/batch-scan", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["results"], data

    best = data["results"][0]["best"]
    rank_score = data["results"][0]["rank_score"]
    base_score = best["annual_savings_value"]
    factor = 1.0 - (payload["risk_penalty"] * (best["risk_score"] / 100.0))

    assert rank_score == pytest.approx(base_score * factor)
