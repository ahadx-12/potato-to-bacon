from pathlib import Path

from potatobacon.tariff.category_detector import CategoryDetector
from potatobacon.tariff.sku_models import SKU
from potatobacon.tariff.sku_store import SKUStore


def test_electronics_detection():
    sku = SKU(
        sku_id="T1",
        description="USB-C charging cable, 6ft",
        declared_value_per_unit=9.99,
        origin_country="VN",
    )
    result = CategoryDetector().detect(sku)
    assert result.primary.name == "electronics"
    assert result.confidence > 0.8


def test_apparel_detection():
    sku = SKU(
        sku_id="T2",
        description="Men's cotton t-shirt, crew neck",
        declared_value_per_unit=6.0,
        origin_country="BD",
    )
    result = CategoryDetector().detect(sku)
    assert result.primary.name == "apparel"


def test_ambiguous_detection():
    sku = SKU(
        sku_id="T3",
        description="Plastic furniture component",
        declared_value_per_unit=2.0,
        origin_country="CN",
    )
    result = CategoryDetector().detect(sku)
    assert len(result.alternatives) >= 2


def test_category_persists_in_registry(tmp_path: Path):
    store = SKUStore(tmp_path / "skus.jsonl")
    record = store.upsert(
        "SKU-IND-PUMP",
        {"description": "Industrial pump assembly", "declared_value_per_unit": 125.0},
    )
    assert record.inferred_category is not None
    assert record.category_confidence is not None
