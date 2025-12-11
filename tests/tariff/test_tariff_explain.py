from potatobacon.tariff.engine import explain_tariff_scenario


def test_tariff_explain_unsat_core_detected():
    facts = {
        "upper_material_textile": True,
        "outer_sole_material_rubber_or_plastics": True,
        "surface_contact_rubber_gt_50": True,
        "surface_contact_textile_gt_50": True,
        "felt_covering_gt_50": False,
    }

    response = explain_tariff_scenario(base_facts=facts)

    assert response.status == "UNSAT"
    assert response.proof_id
    assert any(entry["source_id"] == "HTS_CONTACT_EXCLUSION" for entry in response.unsat_core)
    assert "Conflict" in response.explanation
