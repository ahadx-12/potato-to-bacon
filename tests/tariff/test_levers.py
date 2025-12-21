from potatobacon.tariff.levers import applicable_levers
from potatobacon.tariff.product_schema import ProductCategory, ProductSpecModel


def test_electronics_levers_require_material_and_enclosure():
    spec = ProductSpecModel(product_category=ProductCategory.ELECTRONICS)
    facts = {
        "product_type_electronics": True,
        "electronics_enclosure": True,
        "material_steel": True,
        "electronics_cable_or_connector": True,
        "electronics_has_connectors": True,
    }

    levers = applicable_levers(spec=spec, facts=facts)
    lever_ids = {lever.lever_id for lever in levers}
    assert "ELEC_ENCLOSURE_PLASTIC_DOMINANCE" in lever_ids
    assert "ELEC_CONNECTOR_PATHWAY" in lever_ids
    assert "ELECTRONICS_CABLE_ASSEMBLY_PATHWAY" in lever_ids


def test_apparel_lever_triggers_for_blend_near_threshold():
    spec = ProductSpecModel(product_category=ProductCategory.APPAREL_TEXTILE)
    facts = {
        "product_type_apparel_textile": True,
        "fiber_cotton_dominant": True,
        "fiber_polyester_dominant": False,
        "textile_woven": True,
    }

    levers = applicable_levers(spec=spec, facts=facts)
    lever_ids = {lever.lever_id for lever in levers}
    assert "APPAREL_BLEND_DOMINANCE" in lever_ids
    assert "APPAREL_CONFIRM_KNIT_WOVEN" not in lever_ids


def test_apparel_knit_confirmation_only_when_missing_flags():
    spec = ProductSpecModel(product_category=ProductCategory.APPAREL_TEXTILE)
    facts = {
        "product_type_apparel_textile": True,
    }
    levers = applicable_levers(spec=spec, facts=facts)
    lever_ids = {lever.lever_id for lever in levers}
    assert "APPAREL_CONFIRM_KNIT_WOVEN" in lever_ids
