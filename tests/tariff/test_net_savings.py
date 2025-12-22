import math

from potatobacon.tariff.models import TariffFeasibility
from potatobacon.tariff.optimizer import compute_net_savings_projection
from potatobacon.tariff.models import NetSavings, TariffSuggestionItemModel
from potatobacon.tariff.suggest import _sort_key


def test_net_savings_projection_math():
    feasibility = TariffFeasibility(
        one_time_cost=1000.0,
        recurring_cost_per_unit=0.1,
        implementation_time_days=30,
        requires_recertification=False,
    )
    result = compute_net_savings_projection(
        baseline_rate=2.0,
        optimized_rate=1.0,
        declared_value_per_unit=10.0,
        annual_volume=1000,
        feasibility=feasibility,
    )

    gross_expected = (2.0 - 1.0) / 100.0 * 10.0 * 1000
    first_year_adjustment = (365 - 30) / 365
    first_year_expected = gross_expected * first_year_adjustment
    implementation_cost = 1000 + 0.1 * 1000
    net_expected = first_year_expected - implementation_cost

    assert math.isclose(result.gross_duty_savings, gross_expected, rel_tol=1e-6)
    assert math.isclose(result.first_year_savings, first_year_expected, rel_tol=1e-6)
    assert math.isclose(result.net_annual_savings, net_expected, rel_tol=1e-6)
    assert result.payback_months and result.payback_months > 0


def test_sort_prefers_net_savings_over_rate_delta():
    richer = TariffSuggestionItemModel(
        human_summary="Higher net savings",
        lever_id="L1",
        lever_feasibility="HIGH",
        feasibility=TariffFeasibility(),
        evidence_requirements=[],
        baseline_duty_rate=5.0,
        optimized_duty_rate=1.0,
        savings_per_unit_rate=4.0,
        savings_per_unit_value=4.0,
        annual_savings_value=4000.0,
        net_savings=NetSavings(net_annual_savings=9000.0),
        ranking_score=9000.0,
        best_mutation={},
        classification_confidence=0.8,
        active_codes_baseline=["A"],
        active_codes_optimized=["B"],
        provenance_chain=[],
        law_context="CTX",
        proof_id="p1",
        proof_payload_hash="hash1",
        risk_score=20,
        defensibility_grade="A",
        risk_reasons=[],
        tariff_manifest_hash="mh1",
    )
    leaner = TariffSuggestionItemModel(
        human_summary="Lower net savings",
        lever_id="L2",
        lever_feasibility="HIGH",
        feasibility=TariffFeasibility(),
        evidence_requirements=[],
        baseline_duty_rate=5.0,
        optimized_duty_rate=0.5,
        savings_per_unit_rate=4.5,
        savings_per_unit_value=4.5,
        annual_savings_value=4500.0,
        net_savings=NetSavings(net_annual_savings=1000.0),
        ranking_score=1000.0,
        best_mutation={},
        classification_confidence=0.8,
        active_codes_baseline=["A"],
        active_codes_optimized=["C"],
        provenance_chain=[],
        law_context="CTX",
        proof_id="p2",
        proof_payload_hash="hash2",
        risk_score=20,
        defensibility_grade="A",
        risk_reasons=[],
        tariff_manifest_hash="mh1",
    )

    ordered = sorted(enumerate([richer, leaner]), key=lambda pair: _sort_key(pair[1], 100, pair[0]))
    assert ordered[0][1].proof_id == "p1"
