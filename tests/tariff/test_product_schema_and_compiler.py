from potatobacon.tariff.fact_compiler import compile_facts
from potatobacon.tariff.product_schema import ProductSpecModel
from tests.data.product_specs import bolt_spec, footwear_spec


def test_product_schema_is_deterministic():
    first = ProductSpecModel(**footwear_spec.model_dump())
    second = ProductSpecModel(**footwear_spec.model_dump())
    assert first.model_dump() == second.model_dump()


def test_compile_facts_reproducible():
    facts_one, evidence_one = compile_facts(bolt_spec)
    facts_two, evidence_two = compile_facts(ProductSpecModel(**bolt_spec.model_dump()))
    assert facts_one == facts_two
    assert [e.__dict__ for e in evidence_one] == [e.__dict__ for e in evidence_two]
    assert facts_one["product_category"] == "fastener"
    assert facts_one["material_steel"] is True
    assert any(ev.fact_key == "material_steel" for ev in evidence_one)


