import pytest

from potatobacon.tariff.atoms_hts import DUTY_RATES, tariff_policy_atoms
from potatobacon.tariff.candidate_search import generate_baseline_candidates
from potatobacon.tariff.engine import compute_duty_rate
from potatobacon.tariff.models import TariffScenario


def test_electronics_cable_atoms_prefer_low_voltage_lane():
    atoms = tariff_policy_atoms()
    facts = {
        "product_category": "electronics",
        "product_type_electronics": True,
        "electronics_cable_or_connector": True,
        "electronics_has_connectors": True,
        "electronics_is_cable_assembly": True,
        "electronics_insulated_conductors": True,
        "electronics_voltage_rating_known": True,
    }

    scenario = TariffScenario(name="usb-cable", facts=facts)
    duty_rate = compute_duty_rate(atoms, scenario, strict=False)
    assert duty_rate == pytest.approx(1.0)

    candidates = generate_baseline_candidates(facts, atoms, DUTY_RATES, max_candidates=3)
    assert candidates[0].candidate_id == "HTS_ELECTRONICS_SIGNAL_LOW_VOLT"
    assert candidates[0].missing_facts == []


def test_electronics_harness_lane_requires_insulation():
    atoms = tariff_policy_atoms()
    facts = {
        "product_category": "electronics",
        "product_type_electronics": True,
        "electronics_cable_or_connector": True,
        "electronics_has_connectors": True,
        "electronics_is_cable_assembly": True,
    }

    candidates = generate_baseline_candidates(facts, atoms, DUTY_RATES, max_candidates=5)
    ids = [candidate.candidate_id for candidate in candidates]
    assert "HTS_ELECTRONICS_WIRE_HARNESS" in ids
    harness = next(candidate for candidate in candidates if candidate.candidate_id == "HTS_ELECTRONICS_WIRE_HARNESS")
    assert "electronics_insulated_conductors" in harness.missing_facts
