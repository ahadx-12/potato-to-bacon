import json
from pathlib import Path

from potatobacon.tariff.models import StructuredBOMModel, BOMLineItemModel
from potatobacon.tariff.sku_store import SKUStore


def test_sku_store_upsert_get_list_delete(tmp_path: Path):
    store_path = tmp_path / "skus.jsonl"
    store = SKUStore(store_path)

    record = store.upsert(
        "SKU-1",
        {
            "description": "Canvas shoe with rubber sole",
            "origin_country": "VN",
            "metadata": {"brand": "demo"},
        },
    )
    assert record.created_at is not None
    assert record.updated_at is not None

    fetched = store.get("SKU-1")
    assert fetched is not None
    assert fetched.description == "Canvas shoe with rubber sole"
    assert fetched.created_at == record.created_at

    updated = store.upsert("SKU-1", {"description": "Updated description"})
    assert updated.created_at == record.created_at
    assert updated.updated_at != record.updated_at

    listed = store.list()
    assert [item.sku_id for item in listed] == ["SKU-1"]

    deleted = store.delete("SKU-1")
    assert deleted is True
    assert store.get("SKU-1") is None


def test_sku_store_serialization_and_filters(tmp_path: Path):
    store = SKUStore(tmp_path / "skus.jsonl")
    store.upsert(
        "b-sku",
        {
            "description": "Harness connector assembly",
            "bom_json": StructuredBOMModel(items=[BOMLineItemModel(description="Connector", quantity=2.0)]),
        },
    )
    store.upsert("a-sku", {"description": "Footwear baseline"})

    lines = (tmp_path / "skus.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["sku_id"] for line in lines] == ["a-sku", "b-sku"]

    reload_store = SKUStore(tmp_path / "skus.jsonl")
    assert [item.sku_id for item in reload_store.list()] == ["a-sku", "b-sku"]
    assert reload_store.get("b-sku").bom_json is not None

    assert len(reload_store.list(prefix="a")) == 1
    assert len(reload_store.list(limit=1)) == 1
