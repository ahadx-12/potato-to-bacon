import pytest

from potatobacon.tariff.atoms_hts import DUTY_RATES, tariff_policy_atoms
from potatobacon.tariff.candidate_search import generate_baseline_candidates


def test_candidate_search_ranks_by_rate_and_specificity():
    atoms = tariff_policy_atoms()
    facts = {
        "product_category": "electronics",
        "product_type_electronics": True,
        "electronics_cable_or_connector": True,
        "electronics_has_connectors": True,
        "electronics_is_cable_assembly": True,
    }

    candidates = generate_baseline_candidates(facts, atoms, DUTY_RATES, max_candidates=5)

    assert candidates, "expected duty-bearing candidates"
    # Lower duty atoms should be considered first even if additional facts are needed.
    ids = [cand.candidate_id for cand in candidates[:4]]
    assert ids[0] == "HTS_ELECTRONICS_SIGNAL_LOW_VOLT"
    assert ids[1] == "HTS_ELECTRONICS_ACTIVE_CABLE"
    assert "HTS_ELECTRONICS_CONNECTOR" in ids[:4]
    assert candidates[0].missing_facts  # low-voltage path should request remaining electronics facts


def test_candidate_search_tracks_missing_facts():
    atoms = tariff_policy_atoms()
    facts = {"product_type_apparel_textile": True}

    candidates = generate_baseline_candidates(facts, atoms, DUTY_RATES, max_candidates=3)

    assert candidates, "expected apparel candidates even with missing facts"
    top = candidates[0]
    assert top.missing_facts, "missing fact list should be populated"
    assert 0.0 <= top.confidence <= 1.0
