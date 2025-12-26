from dataclasses import dataclass, field

from potatobacon.tariff.fact_compiler import FactCompiler
from potatobacon.tariff.fact_schema_registry import FactSchemaRegistry
from potatobacon.tariff.sku_models import SKU


def test_fact_schema_for_category():
    schema = FactSchemaRegistry().get_schema_for_category("electronics")
    assert "copper_conductor" in schema
    assert "voltage_rating_v" in schema
    assert len(schema) >= 15


def test_category_facts_applied_to_sku():
    sku = SKU(
        sku_id="S1",
        description="USB cable",
        origin_country="VN",
        declared_value_per_unit=3.0,
        inferred_category="electronics",
        category_confidence=0.9,
    )
    facts = FactSchemaRegistry().get_all_facts_for_sku(sku)
    assert "copper_conductor" in facts
    assert "textile_content_pct" not in facts


@dataclass
class DummySession:
    sku_id: str
    fact_overrides: dict = field(default_factory=dict)


def test_fact_compilation_with_schema():
    sku = SKU(
        sku_id="S2",
        description="USB cable",
        origin_country="VN",
        declared_value_per_unit=3.0,
        inferred_category="electronics",
        category_confidence=0.9,
    )
    session = DummySession(sku_id=sku.sku_id)
    compiled = FactCompiler().compile(sku, session)
    assert isinstance(compiled.facts, dict)
    assert isinstance(compiled.provenance, dict)
    assert all(key in compiled.provenance for key in compiled.facts.keys())
