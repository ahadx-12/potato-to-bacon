import pytest

from potatobacon.tariff.optimizer import optimize_tariff


def test_optimize_tesla_bolt_material_change():
    base_facts = {
        "product_type_chassis_bolt": True,
        "material_steel": True,
        "material_aluminum": False,
    }
    candidate_mutations = {
        "material_steel": [True, False],
        "material_aluminum": [False, True],
    }

    result = optimize_tariff(base_facts, candidate_mutations, law_context=None)

    assert result.baseline_rate == pytest.approx(6.5, rel=1e-6)
    assert result.optimized_rate == pytest.approx(2.5, rel=1e-6)
    assert result.status == "OPTIMIZED"
    assert result.best_mutation in (
        {"material_steel": False},
        {"material_aluminum": True},
        {"material_steel": False, "material_aluminum": True},
    )
    assert result.proof_id
