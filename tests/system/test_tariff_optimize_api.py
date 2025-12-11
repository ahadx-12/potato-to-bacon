import pytest


BASELINE_FACTS = {
    "upper_material_textile": True,
    "outer_sole_material_rubber_or_plastics": True,
    "surface_contact_rubber_gt_50": True,
    "surface_contact_textile_gt_50": False,
    "felt_covering_gt_50": False,
}


@pytest.mark.usefixtures("system_client")
def test_tariff_optimize_endpoint(system_client):
    request = {
        "scenario": BASELINE_FACTS,
        "candidate_mutations": {"felt_covering_gt_50": [False, True]},
    }

    response = system_client.post("/api/tariff/optimize", json=request)
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "OPTIMIZED"
    assert payload["baseline_duty_rate"] == 37.5
    assert payload["optimized_duty_rate"] == 3.0
    assert payload["savings_per_unit"] == 34.5
    assert payload["best_mutation"]["felt_covering_gt_50"] is True
    assert payload["proof_id"]
    assert payload.get("law_context")
