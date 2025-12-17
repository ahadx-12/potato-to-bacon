from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.risk import assess_tariff_risk


def test_fta_flags_present_when_origin_known():
    description = "Knit cotton shirt for men"
    spec, _ = extract_product_spec(
        description,
        bom_text=None,
        origin_country="CA",
        import_country=None,
    )
    facts, _ = compile_facts_with_evidence(spec, description, None)
    assert facts["requires_origin_data"] is False
    assert facts.get("fta_usmca_eligible") is True
    assert facts.get("duty_reduction_possible") is True


def test_ad_cvd_possible_increases_risk():
    description = "Steel chassis bolt fastener"
    spec, _ = extract_product_spec(description, None, origin_country="CN", import_country="US")
    facts, _ = compile_facts_with_evidence(spec, description, None)
    assert facts.get("ad_cvd_possible") is True

    risk = assess_tariff_risk(
        baseline_facts=facts,
        optimized_facts=facts,
        baseline_active_atoms=[],
        optimized_active_atoms=[],
        baseline_duty_rate=6.5,
        optimized_duty_rate=6.5,
    )
    assert any("AD/CVD" in reason for reason in risk.risk_reasons)
    assert risk.risk_score >= 10
