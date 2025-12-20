from potatobacon.tariff.atoms_hts import tariff_policy_atoms
from potatobacon.tariff.engine import compute_duty_result
from potatobacon.tariff.models import TariffScenario


def test_compute_duty_result_handles_missing_rules():
    atoms = tariff_policy_atoms()
    scenario = TariffScenario(name="no-duty-match", facts={"unrelated_fact": True})

    result = compute_duty_result(atoms, scenario)

    assert result.status == "NO_DUTY_RULE_ACTIVE"
    assert result.duty_rate is None
    assert result.duty_atom_ids == []
    assert result.active_atoms == []
