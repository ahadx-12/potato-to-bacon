from potatobacon.law.solver_z3 import analyze_scenario
from potatobacon.tariff.context_loader import (
    get_default_tariff_context,
    get_tariff_atoms_for_context,
)
from potatobacon.tariff.engine import apply_mutations, compute_duty_rate
from potatobacon.tariff.models import TariffScenario
from potatobacon.tariff.mutation_generator import baseline_facts_from_profile
from potatobacon.tariff.risk import assess_tariff_risk


def _build_scenarios(profile_category: str, mutation_patch: dict | None):
    baseline_facts = baseline_facts_from_profile({"category": profile_category})
    baseline = TariffScenario(name="baseline", facts=baseline_facts)
    optimized = apply_mutations(baseline, mutation_patch or {})
    return baseline, optimized


def test_boundary_risk_higher_than_material_swap():
    context = get_default_tariff_context()
    atoms = get_tariff_atoms_for_context(context)

    footwear_baseline, footwear_optimized = _build_scenarios(
        "footwear", {"felt_covering_gt_50": True}
    )
    fastener_baseline, fastener_optimized = _build_scenarios(
        "fastener", {"material_steel": False, "material_aluminum": True}
    )

    footwear_rates = (
        compute_duty_rate(atoms, footwear_baseline),
        compute_duty_rate(atoms, footwear_optimized),
    )
    fastener_rates = (
        compute_duty_rate(atoms, fastener_baseline),
        compute_duty_rate(atoms, fastener_optimized),
    )

    _, footwear_baseline_atoms, _ = analyze_scenario(footwear_baseline.facts, atoms)
    _, footwear_optimized_atoms, _ = analyze_scenario(footwear_optimized.facts, atoms)

    _, fastener_baseline_atoms, _ = analyze_scenario(fastener_baseline.facts, atoms)
    _, fastener_optimized_atoms, _ = analyze_scenario(fastener_optimized.facts, atoms)

    footwear_risk = assess_tariff_risk(
        baseline_facts=footwear_baseline.facts,
        optimized_facts=footwear_optimized.facts,
        baseline_active_atoms=footwear_baseline_atoms,
        optimized_active_atoms=footwear_optimized_atoms,
        baseline_duty_rate=footwear_rates[0],
        optimized_duty_rate=footwear_rates[1],
    )
    fastener_risk = assess_tariff_risk(
        baseline_facts=fastener_baseline.facts,
        optimized_facts=fastener_optimized.facts,
        baseline_active_atoms=fastener_baseline_atoms,
        optimized_active_atoms=fastener_optimized_atoms,
        baseline_duty_rate=fastener_rates[0],
        optimized_duty_rate=fastener_rates[1],
    )

    assert footwear_risk.risk_score > fastener_risk.risk_score
    assert "threshold" in " ".join(footwear_risk.risk_reasons).lower()
    assert "threshold" not in " ".join(fastener_risk.risk_reasons).lower()
    assert footwear_risk.defensibility_grade in {"B", "C"}
    assert fastener_risk.defensibility_grade in {"A", "B"}


def test_assess_tariff_risk_is_deterministic():
    context = get_default_tariff_context()
    atoms = get_tariff_atoms_for_context(context)

    baseline, optimized = _build_scenarios("footwear", {"felt_covering_gt_50": True})
    baseline_rate = compute_duty_rate(atoms, baseline)
    optimized_rate = compute_duty_rate(atoms, optimized)
    _, baseline_atoms, _ = analyze_scenario(baseline.facts, atoms)
    _, optimized_atoms, _ = analyze_scenario(optimized.facts, atoms)

    first = assess_tariff_risk(
        baseline_facts=baseline.facts,
        optimized_facts=optimized.facts,
        baseline_active_atoms=baseline_atoms,
        optimized_active_atoms=optimized_atoms,
        baseline_duty_rate=baseline_rate,
        optimized_duty_rate=optimized_rate,
    )
    second = assess_tariff_risk(
        baseline_facts=baseline.facts,
        optimized_facts=optimized.facts,
        baseline_active_atoms=baseline_atoms,
        optimized_active_atoms=optimized_atoms,
        baseline_duty_rate=baseline_rate,
        optimized_duty_rate=optimized_rate,
    )

    assert first == second
