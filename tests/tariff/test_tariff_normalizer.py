from potatobacon.tariff.normalizer import normalize_compiled_facts, validate_minimum_inputs


def test_normalizer_resolves_conflicts():
    facts = {
        "surface_contact_textile_gt_50": True,
        "surface_contact_rubber_gt_50": True,
        "material_steel": True,
        "material_aluminum": True,
        "textile_knit": True,
        "textile_woven": True,
    }

    normalized, notes = normalize_compiled_facts(facts)

    assert normalized["surface_contact_textile_gt_50"] is True
    assert normalized["surface_contact_rubber_gt_50"] is False
    assert normalized["requires_measurement"] is True
    assert normalized["textile_knit"] is True
    assert normalized["textile_woven"] is False
    assert notes


def test_validate_minimum_inputs_detects_missing():
    product_spec = {"product_category": "ELECTRONICS"}
    facts = {}

    missing = validate_minimum_inputs(product_spec, facts)

    assert "electronics form factor" in missing
    assert "dominant material" in missing
