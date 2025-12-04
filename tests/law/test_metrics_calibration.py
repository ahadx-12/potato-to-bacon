import random

from potatobacon.law.cale_metrics import compute_scenario_metrics
from potatobacon.law.solver_z3 import PolicyAtom


def _make_atom(jurisdiction: str, action: str, guard=None, modality: str = "OBLIGE") -> PolicyAtom:
    return PolicyAtom(
        guard=guard or [],
        outcome={"modality": modality, "action": action, "jurisdiction": jurisdiction},
        source_id=action,
    )


def test_risk_components_clamped_and_net_after_tax_positive():
    rng = random.Random(42)
    scenarios = []
    for idx in range(3):
        flag = bool(rng.getrandbits(1))
        atoms = [
            _make_atom("US", f"pay_tax_{idx}", guard=["p" if flag else "¬p"]),
            _make_atom("IE", f"deduct_{idx}", modality="PERMIT"),
            _make_atom("KY", f"withhold_{idx}", modality="FORBID"),
        ]
        scenario = {"p": flag, f"flag_{idx}": True}
        metrics = compute_scenario_metrics(scenario, atoms, seed=idx)
        scenarios.append(metrics)

        for value in metrics.risk_components.values():
            assert 0.0 <= value <= 1.0
        assert 0.0 <= metrics.risk <= 1.0

        gross_income = metrics.value_components["gross_income"]
        tax_liability = metrics.value_components["tax_liability"]
        if gross_income > tax_liability:
            assert metrics.value_components["net_after_tax"] > 0.0

    diverse = [m.contradiction_probability for m in scenarios]
    assert all(prob < 1.0 for prob in diverse)
