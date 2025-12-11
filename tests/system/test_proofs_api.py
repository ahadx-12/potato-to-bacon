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
def test_proofs_api_returns_saved_record(system_client):
    request = TariffHuntRequestModel(
        scenario=BASELINE_FACTS,
        mutations={"felt_covering_gt_50": True},
        seed=2025,
    )

    response = system_client.post("/api/tariff/analyze", json=request.model_dump())
    assert response.status_code == 200, response.text
    proof_id = response.json()["proof_id"]

    proof_response = system_client.get(f"/v1/proofs/{proof_id}")
    assert proof_response.status_code == 200, proof_response.text
    proof_payload = proof_response.json()
    assert proof_payload["proof_id"] == proof_id
    assert proof_payload["input"]["scenario"]["felt_covering_gt_50"] is False
