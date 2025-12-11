import pytest

from potatobacon.tariff.models import TariffHuntRequestModel


BASELINE_FACTS = {
    "upper_material_textile": True,
    "outer_sole_material_rubber_or_plastics": True,
    "surface_contact_rubber_gt_50": True,
    "surface_contact_textile_gt_50": False,
    "felt_covering_gt_50": False,
}


@pytest.mark.usefixtures("system_client")
def test_tariff_api_analyze_endpoint(system_client):
    request = TariffHuntRequestModel(
        scenario=BASELINE_FACTS,
        mutations={"felt_covering_gt_50": True},
        seed=2025,
    )

    response = system_client.post("/api/tariff/analyze", json=request.model_dump())
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "OPTIMIZED"
    assert payload["proof_id"]
    assert payload.get("law_context")
    assert payload["baseline_duty_rate"] == 37.5
    assert payload["optimized_duty_rate"] == 3.0
    assert payload["savings_per_unit"] == 34.5
    assert payload["baseline_scenario"]["felt_covering_gt_50"] is False
    assert payload["optimized_scenario"]["felt_covering_gt_50"] is True
    assert payload["optimized_scenario"]["surface_contact_textile_gt_50"] is True
    assert payload["optimized_scenario"]["surface_contact_rubber_gt_50"] is False
    assert "HTS_6404_11_90" in payload["active_codes_baseline"]
    assert "HTS_6404_19_35" in payload["active_codes_optimized"]

    provenance_ids = {entry.get("source_id") for entry in payload.get("provenance_chain", [])}
    assert "HTS_6404_11_90" in provenance_ids
    assert "HTS_6404_19_35" in provenance_ids
