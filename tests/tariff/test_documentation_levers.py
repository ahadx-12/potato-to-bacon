from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.candidate_search import generate_baseline_candidates
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.normalizer import normalize_compiled_facts
from potatobacon.tariff.suggest import DOCUMENTATION_LEVER_ID, _documentation_levers, _evaluate_scenario


def test_documentation_lever_unlocks_low_voltage_pathway():
    atoms, context_meta = load_atoms_for_context(DEFAULT_CONTEXT_ID)
    duty_rates = context_meta.get("duty_rates") or DUTY_RATES
    baseline_facts = {
        "product_type_electronics": True,
        "electronics_cable_or_connector": True,
        "electronics_has_connectors": True,
        "electronics_is_cable_assembly": True,
        "electronics_voltage_rating_known": True,
        "origin_country": "VN",
        "import_country": "US",
    }
    normalized_facts, _ = normalize_compiled_facts(baseline_facts)
    overlay_context = {
        "origin_country": "VN",
        "import_country": "US",
        "hts_code": normalized_facts.get("hts_code"),
    }

    baseline_candidates = generate_baseline_candidates(normalized_facts, atoms, duty_rates, max_candidates=5)
    baseline_eval = _evaluate_scenario(atoms, normalized_facts, duty_rates, overlay_context=overlay_context)
    baseline_rate = baseline_eval.effective_duty_rate
    baseline_confidence = baseline_candidates[0].confidence if baseline_candidates else 0.3

    documentation_levers = _documentation_levers(
        baseline_candidates=baseline_candidates,
        atoms=atoms,
        duty_rates=duty_rates,
        baseline_eval=baseline_eval,
        baseline_facts=normalized_facts,
        baseline_rate=baseline_rate,
        baseline_rate_raw=baseline_eval.duty_rate,
        baseline_confidence=baseline_confidence,
        declared_value=10.0,
        annual_volume=100000,
        law_context=context_meta["context_id"],
        context_meta=context_meta,
        evidence_pack={},
        overlay_context=overlay_context,
    )

    assert documentation_levers, "Documentation lever should be returned"
    doc_lever = next(lever for lever in documentation_levers if lever.target_candidate == "HTS_ELECTRONICS_SIGNAL_LOW_VOLT")
    assert doc_lever.lever_category == DOCUMENTATION_LEVER_ID
    assert doc_lever.optimization_type == "CONDITIONAL_OPTIMIZATION"
    assert doc_lever.optimized_duty_rate < (baseline_rate or 999)
    assert doc_lever.savings_per_unit_rate > 0
    assert doc_lever.fact_gaps == ["electronics_insulated_conductors"]
    expected_templates = [
        "bom_csv",
        "harness_cross_section_photo",
        "lab_test_report",
        "manufacturer_datasheet_pdf",
        "material_declaration",
        "product_photo_label",
        "spec_sheet",
    ]
    assert doc_lever.accepted_evidence_templates == sorted(expected_templates)
