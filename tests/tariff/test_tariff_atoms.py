from potatobacon.tariff.atoms_hts import tariff_policy_atoms
from potatobacon.tariff.engine import apply_mutations, compute_duty_rate
from potatobacon.tariff.models import TariffScenario


BASELINE_FACTS = {
    "upper_material_textile": True,
    "outer_sole_material_rubber_or_plastics": True,
    "surface_contact_rubber_gt_50": True,
    "surface_contact_textile_gt_50": False,
    "felt_covering_gt_50": False,
}


def test_tariff_atoms_baseline_and_optimized_rates():
    atoms = tariff_policy_atoms()
    baseline = TariffScenario(name="converse-baseline", facts=BASELINE_FACTS)

    baseline_rate = compute_duty_rate(atoms, baseline)
    assert baseline_rate == 37.5

    optimized = apply_mutations(baseline, {"felt_covering_gt_50": True})
    assert optimized.facts["felt_covering_gt_50"] is True
    assert optimized.facts["surface_contact_textile_gt_50"] is True
    assert optimized.facts["surface_contact_rubber_gt_50"] is False

    optimized_rate = compute_duty_rate(atoms, optimized)
    assert optimized_rate == 3.0
    assert optimized_rate < baseline_rate
