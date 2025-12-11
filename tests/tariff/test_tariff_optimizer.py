import pytest

from potatobacon.tariff.optimizer import optimize_tariff


def test_optimize_converse_felt_overlay():
    base_facts = {
        "upper_material_textile": True,
        "outer_sole_material_rubber_or_plastics": True,
        "surface_contact_rubber_gt_50": True,
        "surface_contact_textile_gt_50": False,
        "felt_covering_gt_50": False,
    }
    candidate_mutations = {"felt_covering_gt_50": [False, True]}

    result = optimize_tariff(base_facts, candidate_mutations, law_context=None)

    assert result.baseline_rate == pytest.approx(37.5, rel=1e-6)
    assert result.optimized_rate == pytest.approx(3.0, rel=1e-6)
    assert result.status == "OPTIMIZED"
    assert result.best_mutation == {"felt_covering_gt_50": True}
    assert result.proof_id
