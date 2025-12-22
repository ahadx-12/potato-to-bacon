import pytest

from potatobacon.tariff.engine import compute_duty_rate
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.models import TariffScenario


BASELINE_FACTS = {
    "upper_material_textile": True,
    "outer_sole_material_rubber_or_plastics": True,
    "surface_contact_rubber_gt_50": True,
    "surface_contact_textile_gt_50": False,
    "felt_covering_gt_50": False,
    "origin_country": "CN",
}


@pytest.mark.usefixtures("system_client")
def test_section_301_overlays_recorded_in_proof(system_client):
    response = system_client.post("/api/tariff/analyze", json={"scenario": BASELINE_FACTS})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "REQUIRES_REVIEW"
    assert payload["overlays"]["baseline"], "Expected baseline overlays to be present"

    baseline_overlay = payload["overlays"]["baseline"][0]
    assert baseline_overlay["overlay_name"].startswith("Section 301")
    assert baseline_overlay["stop_optimization"] is True
    assert pytest.approx(payload["baseline_duty_rate"] + baseline_overlay["additional_rate"]) == payload[
        "baseline_effective_duty_rate"
    ]

    proof_id = payload["proof_id"]
    proof_response = system_client.get(f"/v1/proofs/{proof_id}")
    assert proof_response.status_code == 200, proof_response.text
    proof_payload = proof_response.json()
    proof_overlays = proof_payload.get("overlays") or {}
    assert proof_overlays.get("baseline")

    atoms, _ = load_atoms_for_context(DEFAULT_CONTEXT_ID)
    baseline = TariffScenario(name="baseline", facts=payload["baseline_scenario"])
    baseline_rate = compute_duty_rate(atoms, baseline)
    proof_baseline_overlay = proof_overlays["baseline"][0]
    expected_effective = baseline_rate + proof_baseline_overlay["additional_rate"]
    assert pytest.approx(expected_effective) == payload["baseline_effective_duty_rate"]
