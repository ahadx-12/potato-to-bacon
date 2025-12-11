from potatobacon.tariff.engine import run_tariff_hack
from potatobacon.tariff.models import TariffHuntRequestModel


BASELINE_FACTS = {
    "upper_material_textile": True,
    "outer_sole_material_rubber_or_plastics": True,
    "surface_contact_rubber_gt_50": True,
    "surface_contact_textile_gt_50": False,
    "felt_covering_gt_50": False,
}


def test_run_tariff_hack_returns_expected_dossier():
    request = TariffHuntRequestModel(
        scenario=BASELINE_FACTS,
        mutations={"felt_covering_gt_50": True},
        seed=2025,
    )
    dossier = run_tariff_hack(
        base_facts=request.scenario,
        mutations=request.mutations,
        law_context=request.law_context,
        seed=request.seed or 2025,
    )

    assert dossier.status == "OPTIMIZED"
    assert dossier.proof_id
    assert dossier.law_context
    assert dossier.baseline_duty_rate == 37.5
    assert dossier.optimized_duty_rate == 3.0
    assert dossier.savings_per_unit == 34.5
    assert dossier.baseline_scenario["felt_covering_gt_50"] is False
    assert dossier.optimized_scenario["felt_covering_gt_50"] is True
    assert dossier.optimized_scenario["surface_contact_textile_gt_50"] is True
    assert dossier.optimized_scenario["surface_contact_rubber_gt_50"] is False
    assert dossier.active_codes_baseline == ["HTS_6404_11_90"]
    assert dossier.active_codes_optimized[-1] == "HTS_6404_19_35"
