from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.product_schema import ProductCategory


def test_parse_canvas_sneaker():
    description = "Canvas sneaker with rubber sole felt overlay"

    spec, extraction_evidence = extract_product_spec(description, None)
    assert spec.product_category == ProductCategory.FOOTWEAR
    assert any("sneaker" in ev.snippet.lower() for ev in extraction_evidence)

    facts, fact_evidence = compile_facts_with_evidence(spec, description, None)
    assert facts.get("material_textile") is True
    assert facts.get("surface_contact_rubber_gt_50") is True
    assert any(ev.evidence for ev in fact_evidence)
    assert any("felt" in (ev.snippet.lower()) for fev in fact_evidence for ev in fev.evidence)


def test_parse_chassis_bolt():
    description = "Chassis bolt fastener made of steel"

    spec, extraction_evidence = extract_product_spec(description, None)
    assert spec.product_category == ProductCategory.FASTENER
    assert any("bolt" in ev.snippet.lower() for ev in extraction_evidence)

    facts, fact_evidence = compile_facts_with_evidence(spec, description, None)
    assert facts.get("product_type_chassis_bolt") is True
    assert facts.get("material_steel") is True
    assert any("steel" in ev.snippet.lower() for fev in fact_evidence for ev in fev.evidence)


def test_parse_unknown_description():
    description = "Miscellaneous item with unclear details"

    spec, _ = extract_product_spec(description, None)
    assert spec.product_category == ProductCategory.OTHER

    _, fact_evidence = compile_facts_with_evidence(spec, description, None)
    # product_category fact exists but carries low confidence for unknown category
    confidences = [fev.confidence for fev in fact_evidence if fev.fact_key == "product_category"]
    assert confidences and confidences[0] <= 0.4
