from __future__ import annotations

import csv
from io import StringIO
from typing import Any, Dict, List

from .models import BOMLineItemModel, StructuredBOMModel

SUPPORTED_HEADERS = {
    "part_id": "part_id",
    "description": "description",
    "material": "material",
    "quantity": "quantity",
    "unit_cost": "unit_cost",
    "weight_kg": "weight_kg",
    "intended_use": "intended_use",
    "country_of_origin": "country_of_origin",
    "hts_code": "hts_code",
}


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bom_csv(csv_text: str) -> StructuredBOMModel:
    """Parse supported CSV headers into a :class:`StructuredBOMModel`."""

    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        return StructuredBOMModel(items=[], currency="USD")

    normalized_fields = [field.strip().lower() for field in reader.fieldnames]
    header_map = {name: SUPPORTED_HEADERS.get(name, name) for name in normalized_fields}

    items: List[BOMLineItemModel] = []
    for row in reader:
        normalized_row: Dict[str, str | None] = {}
        for raw_key, raw_value in row.items():
            key = header_map.get(raw_key.strip().lower())
            if key in SUPPORTED_HEADERS.values():
                normalized_row[key] = raw_value.strip() if isinstance(raw_value, str) else raw_value

        description = normalized_row.get("description") or ""
        item = BOMLineItemModel(
            part_id=normalized_row.get("part_id"),
            description=description,
            material=normalized_row.get("material"),
            quantity=_coerce_float(normalized_row.get("quantity")),
            unit_cost=_coerce_float(normalized_row.get("unit_cost")),
            weight_kg=_coerce_float(normalized_row.get("weight_kg")),
            intended_use=normalized_row.get("intended_use"),
            country_of_origin=normalized_row.get("country_of_origin"),
            hts_code=normalized_row.get("hts_code"),
        )
        items.append(item)

    return StructuredBOMModel(items=items)


def bom_to_text(bom: StructuredBOMModel) -> str:
    """Render a structured BOM into deterministic text for keyword extraction."""

    lines: List[str] = []
    for idx, item in enumerate(bom.items):
        parts: List[str] = [f"item{idx}", f"description={item.description}"]
        if item.part_id:
            parts.append(f"part={item.part_id}")
        if item.material:
            parts.append(f"material={item.material}")
        if item.country_of_origin:
            parts.append(f"origin={item.country_of_origin}")
        if item.hts_code:
            parts.append(f"hts={item.hts_code}")
        if item.intended_use:
            parts.append(f"use={item.intended_use}")
        if item.weight_kg is not None:
            parts.append(f"weight_kg={item.weight_kg}")
        if item.quantity is not None:
            parts.append(f"qty={item.quantity}")
        if item.unit_cost is not None:
            parts.append(f"cost={item.unit_cost}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def bom_aggregate_material_signals(bom: StructuredBOMModel) -> Dict[str, Any]:
    """Aggregate BOM-level material and origin signals deterministically."""

    material_counts: Dict[str, int] = {}
    origin_counts: Dict[str, int] = {}

    for item in bom.items:
        if item.material:
            key = item.material.lower()
            material_counts[key] = material_counts.get(key, 0) + 1
        if item.country_of_origin:
            origin_key = item.country_of_origin.upper()
            origin_counts[origin_key] = origin_counts.get(origin_key, 0) + 1

    dominant_material = None
    if material_counts:
        dominant_material = sorted(material_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]

    primary_origin = None
    if origin_counts:
        primary_origin = sorted(origin_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]

    return {
        "material_counts": material_counts,
        "origin_counts": origin_counts,
        "dominant_material": dominant_material,
        "primary_origin": primary_origin,
    }
