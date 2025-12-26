from __future__ import annotations

from typing import Dict

from potatobacon.tariff.fact_schemas import (
    APPAREL_FACTS,
    ELECTRONICS_FACTS,
    FURNITURE_FACTS,
    MACHINERY_FACTS,
    PLASTICS_FACTS,
    FactDef,
)
from potatobacon.tariff.sku_models import SKU


BASE_FACTS: Dict[str, FactDef] = {
    "origin_country": FactDef(type="categorical"),
    "export_country": FactDef(type="categorical"),
    "import_country": FactDef(type="categorical"),
    "declared_value_per_unit": FactDef(type="numeric", unit="USD"),
    "annual_volume": FactDef(type="numeric"),
    "current_hts": FactDef(type="categorical"),
    "inferred_category": FactDef(type="categorical"),
}


class FactSchemaRegistry:
    def get_schema_for_category(self, category: str) -> Dict[str, FactDef]:
        category_key = (category or "").lower()
        if category_key == "electronics":
            return dict(ELECTRONICS_FACTS)
        if category_key == "apparel":
            return dict(APPAREL_FACTS)
        if category_key == "machinery":
            return dict(MACHINERY_FACTS)
        if category_key == "furniture":
            return dict(FURNITURE_FACTS)
        if category_key == "plastics":
            return dict(PLASTICS_FACTS)
        return {}

    def get_all_facts_for_sku(self, sku: SKU) -> Dict[str, FactDef]:
        category = getattr(sku, "inferred_category", None)
        category_schema = self.get_schema_for_category(category or "")
        merged = dict(BASE_FACTS)
        merged.update(category_schema)
        return merged
