import pytest

from potatobacon.tariff.models import TariffHuntRequestModel


@pytest.mark.usefixtures("system_client")
def test_tariff_explain_api_unsat(system_client):
    request = TariffHuntRequestModel(
        scenario={
            "upper_material_textile": True,
            "outer_sole_material_rubber_or_plastics": True,
            "surface_contact_rubber_gt_50": True,
            "surface_contact_textile_gt_50": True,
            "felt_covering_gt_50": False,
        }
    )

    response = system_client.post("/v1/tariff/explain", json=request.model_dump())
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "UNSAT"
    assert payload["proof_id"]
    assert any(entry["source_id"] == "HTS_CONTACT_EXCLUSION" for entry in payload["unsat_core"])
    assert "Conflict" in payload["explanation"]
