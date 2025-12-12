import pytest


BASELINE_FACTS = {
    "product_type_chassis_bolt": True,
    "material_steel": True,
    "material_aluminum": False,
}


@pytest.mark.usefixtures("system_client")
def test_tariff_sku_optimize_endpoint(system_client):
    request = {
        "sku_id": "TESLA_BOLT_X1",
        "description": "Chassis bolt for EV frame",
        "scenario": BASELINE_FACTS,
        "candidate_mutations": {
            "material_steel": [True, False],
            "material_aluminum": [False, True],
        },
        "declared_value_per_unit": 200.0,
        "annual_volume": 50000,
    }

    response = system_client.post("/api/tariff/sku-optimize", json=request)
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "OPTIMIZED"
    assert payload["baseline_duty_rate"] == pytest.approx(6.5, rel=1e-6)
    assert payload["optimized_duty_rate"] == pytest.approx(2.5, rel=1e-6)
    assert payload["savings_per_unit_rate"] == pytest.approx(4.0, rel=1e-6)
    assert payload["savings_per_unit_value"] == pytest.approx(8.0, rel=1e-6)
    assert payload["annual_savings_value"] == pytest.approx(400000.0, rel=1e-6)
    assert payload["proof_id"]
