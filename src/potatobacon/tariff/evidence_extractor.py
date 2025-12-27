from __future__ import annotations

"""Structured evidence extraction for tariff workflows."""

import csv
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

from potatobacon.tariff.models import ProductGraph, ProductGraphComponent, ProductOperation


_INSULATION_MATERIALS = {"plastic", "pvc", "rubber", "tpe", "insulation", "jacket"}
_CABLE_KEYWORDS = {"cable", "wire", "harness", "connector", "usb", "hdmi"}
_INSULATION_KEYWORDS = {"insulated", "jacketed", "shielded", "double insulated", "braided"}
_VOLTAGE_KEYWORDS = {"5v", "12v", "24v", "low voltage", "voltage", "usb", "hdmi"}


@dataclass(frozen=True)
class EvidenceExtractionResult:
    """Normalized extraction payload from any evidence blob."""

    kind: Optional[str]
    product_graph: Optional[ProductGraph]
    extracted_facts: Dict[str, Any]


def _normalize_text(content: bytes, content_type: str) -> str:
    try:
        return content.decode("utf-8")
    except Exception:
        try:
            return content.decode("latin-1")
        except Exception:
            return ""


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "_")


def _row_value(normalized_row: Dict[str, str | None]) -> float:
    quantity_text = normalized_row.get("quantity") or "1"
    unit_cost_text = normalized_row.get("unit_cost")
    value_text = normalized_row.get("value")
    try:
        quantity = float(quantity_text) if quantity_text not in {None, ""} else 1.0
    except ValueError:
        quantity = 1.0
    for candidate in (value_text, unit_cost_text):
        if candidate in {None, ""}:
            continue
        try:
            return float(candidate) * quantity
        except ValueError:
            continue
    return quantity


def _component_name(row: Dict[str, str | None], fallback_idx: int) -> str:
    for field in ("part_name", "part_id", "component", "name", "description"):
        val = row.get(field)
        if val:
            return str(val).strip()
    return f"component_{fallback_idx}"


def _normalize_origin(raw_origin: Optional[str]) -> Optional[str]:
    if not raw_origin:
        return None
    return str(raw_origin).strip().upper()


def _parse_bool(raw_value: Optional[str]) -> Optional[bool]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    value = str(raw_value).strip().lower()
    if value in {"yes", "true", "1", "y", "originating"}:
        return True
    if value in {"no", "false", "0", "n", "non-originating"}:
        return False
    return None


def _bom_csv_to_product_graph(csv_text: str) -> Tuple[ProductGraph, Dict[str, Any]]:
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        return ProductGraph(), {}

    normalized_headers = [_normalize_header(field) for field in reader.fieldnames]
    header_map = {field: _normalize_header(field) for field in reader.fieldnames}
    value_fields = {"unit_cost", "value", "value_share"}
    material_fields = {"material", "material_name"}
    origin_fields = {"origin", "origin_country", "country_of_origin"}
    hts_fields = {"hts", "hts_code", "tariff_code", "hs_code", "hs"}
    originating_fields = {"is_originating_material", "originating_material", "originating"}
    function_fields = {"function", "role"}
    name_fields = {"part_name", "component", "name", "description", "part_id"}
    operation_fields = {"operation", "process", "manufacturing_step", "step"}
    operation_country_fields = {"operation_country", "process_country", "country_of_operation"}
    recognized = (
        value_fields
        | material_fields
        | origin_fields
        | hts_fields
        | originating_fields
        | function_fields
        | name_fields
        | operation_fields
        | operation_country_fields
        | {"quantity"}
    )

    components: List[ProductGraphComponent] = []
    operations: List[ProductOperation] = []
    total_value = 0.0
    staged_rows: List[Dict[str, str | None]] = []

    for raw_row in reader:
        normalized_row: Dict[str, str | None] = {}
        for raw_key, raw_val in raw_row.items():
            key = header_map.get(raw_key, _normalize_header(raw_key))
            if key in recognized:
                normalized_row[key] = raw_val.strip() if isinstance(raw_val, str) else raw_val
        staged_rows.append(normalized_row)
        total_value += max(_row_value(normalized_row), 0.0)

    extracted_facts: Dict[str, Any] = {}
    for idx, row in enumerate(staged_rows):
        name = _component_name(row, idx)
        material = row.get("material") or row.get("material_name")
        origin = _normalize_origin(row.get("origin") or row.get("origin_country") or row.get("country_of_origin"))
        hts_code = row.get("hts") or row.get("hts_code") or row.get("tariff_code") or row.get("hs_code") or row.get("hs")
        originating_flag = _parse_bool(
            row.get("is_originating_material") or row.get("originating_material") or row.get("originating")
        )
        function = row.get("function") or row.get("role")
        component_value = _row_value(row)
        value_raw = row.get("value_share")
        share: float | None = None
        if value_raw:
            try:
                share_val = float(value_raw)
                share = share_val / 100.0 if share_val > 1 else share_val
            except ValueError:
                share = None
        if share is None:
            share = (component_value / total_value) if total_value > 0 else None

        comp = ProductGraphComponent(
            name=name,
            material=material,
            hts_code=hts_code,
            value_share=share,
            component_value=component_value,
            origin_country=origin,
            is_originating_material=originating_flag,
            function=function,
        )
        components.append(comp)

        operation = row.get("operation") or row.get("process") or row.get("manufacturing_step") or row.get("step")
        operation_country = row.get("operation_country") or row.get("process_country") or row.get("country_of_operation")
        if operation:
            operations.append(
                ProductOperation(
                    step=str(operation).strip(),
                    country=_normalize_origin(operation_country),
                )
            )

        if material:
            fact_key = f"material_{material.strip().lower()}"
            extracted_facts[fact_key] = True
        if origin:
            extracted_facts[f"origin_country_{origin}"] = True
        lowered_desc = (row.get("description") or "").lower()
        lowered_name = name.lower()
        if any(keyword in lowered_desc or keyword in lowered_name for keyword in _CABLE_KEYWORDS):
            extracted_facts.setdefault("electronics_is_cable_assembly", True)
            extracted_facts.setdefault("electronics_has_connectors", True)
        if material and material.strip().lower() in _INSULATION_MATERIALS:
            extracted_facts["electronics_insulated_conductors"] = True

    components.sort(key=lambda comp: comp.name.lower())
    operations = sorted({op.step.lower(): op for op in operations}.values(), key=lambda op: op.step.lower())
    product_graph = ProductGraph(
        components=components,
        ops=operations,
        attributes={"headers": normalized_headers},
    )
    if extracted_facts.get("electronics_insulated_conductors"):
        extracted_facts.setdefault("electronics_insulation_documented", True)
    return product_graph, extracted_facts


def _spec_sheet_to_facts(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lowered = text.lower()
    attrs: Dict[str, Any] = {}
    facts: Dict[str, Any] = {}
    for keyword in _INSULATION_KEYWORDS:
        if keyword in lowered:
            facts["electronics_insulated_conductors"] = True
            attrs.setdefault("insulation_terms", []).append(keyword)
    for keyword in _VOLTAGE_KEYWORDS:
        if keyword in lowered:
            facts["electronics_voltage_rating_known"] = True
            attrs.setdefault("voltage_terms", []).append(keyword)
    for keyword in _CABLE_KEYWORDS:
        if keyword in lowered:
            facts["electronics_is_cable_assembly"] = True
            facts["electronics_has_connectors"] = True
            attrs.setdefault("connector_terms", []).append(keyword)
    if facts.get("electronics_insulated_conductors"):
        facts.setdefault("electronics_insulation_documented", True)
    return attrs, facts


def _resolve_kind(evidence_kind: Optional[str], content_type: str) -> Optional[str]:
    if evidence_kind:
        return evidence_kind
    normalized = content_type.lower()
    if "csv" in normalized:
        return "bom_csv"
    if "text" in normalized or "json" in normalized:
        return "spec_sheet"
    return None


def extract_evidence(
    content: bytes, *, content_type: str, evidence_kind: str | None = None, filename: str | None = None
) -> EvidenceExtractionResult:
    """Best-effort extraction of structured facts and product graph."""

    resolved_kind = _resolve_kind(evidence_kind, content_type)
    if resolved_kind == "bom_csv":
        text = _normalize_text(content, content_type)
        product_graph, facts = _bom_csv_to_product_graph(text)
        return EvidenceExtractionResult(kind=resolved_kind, product_graph=product_graph, extracted_facts=facts)

    if resolved_kind == "spec_sheet":
        text = _normalize_text(content, content_type)
        attrs, facts = _spec_sheet_to_facts(text)
        product_graph = ProductGraph(attributes=attrs)
        return EvidenceExtractionResult(kind=resolved_kind, product_graph=product_graph, extracted_facts=facts)

    return EvidenceExtractionResult(kind=resolved_kind, product_graph=None, extracted_facts={})
