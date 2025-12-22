from potatobacon.tariff.evidence_extractor import extract_evidence


def test_bom_csv_extraction_builds_product_graph_and_facts():
    csv_text = """part_name,material,origin_country,value
connector shell,plastic,VN,1.2
copper core,copper,VN,1.5
"""
    result = extract_evidence(csv_text.encode("utf-8"), content_type="text/csv", evidence_kind="bom_csv")

    assert result.kind == "bom_csv"
    assert result.product_graph is not None
    components = result.product_graph.components
    assert any(comp.name.startswith("connector") for comp in components)
    assert any(comp.value_share is not None for comp in components)
    assert result.extracted_facts["origin_country_VN"] is True
    assert result.extracted_facts["electronics_insulated_conductors"] is True


def test_spec_sheet_text_flags_voltage_and_insulation():
    text = "USB harness with braided jacket, shielded conductors, rated 5V"
    result = extract_evidence(text.encode("utf-8"), content_type="text/plain", evidence_kind="spec_sheet")

    assert result.kind == "spec_sheet"
    assert result.product_graph is not None
    assert result.extracted_facts["electronics_voltage_rating_known"] is True
    assert result.extracted_facts["electronics_insulated_conductors"] is True
