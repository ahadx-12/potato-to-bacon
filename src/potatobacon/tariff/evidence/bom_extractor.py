from __future__ import annotations

import csv
import importlib.util
import io
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from potatobacon.tariff.evidence.types import ExtractedEvidence
from potatobacon.tariff.evidence_store import EvidenceStore


COLUMN_MAP: Dict[str, List[str]] = {
    "material": ["material", "mat", "material_type", "material_desc"],
    "description": ["description", "desc", "part_description", "component", "part_name"],
    "quantity": ["quantity", "qty", "count", "units", "unit_qty"],
    "unit_price": ["unit_price", "price", "cost", "unit_cost", "price_per_unit"],
    "total_value": ["total_value", "total", "extended_price", "total_price", "ext_price"],
    "origin": ["origin", "country", "country_of_origin", "coo", "country_origin"],
    "supplier": ["supplier", "vendor", "manufacturer", "mfr"],
    "hts": ["hts", "hts_code", "tariff_code", "hs_code", "hs", "harmonized"],
}

_TEXTILE_MATERIALS = [
    "cotton",
    "polyester",
    "nylon",
    "wool",
    "silk",
    "rayon",
    "spandex",
    "elastane",
    "acrylic",
    "viscose",
]

_SUPPORTED_CONTENT_TYPES = {
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sorted_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: data[key] for key in sorted(data.keys())}


def _normalize_space(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalize_column(value: str) -> str:
    cleaned = _normalize_space(value).lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    return cleaned.strip("_")


def _normalize_material(value: str) -> str:
    return _normalize_space(value).lower()


def _normalize_origin(value: str) -> str:
    return _normalize_space(value)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


class BOMExtractor:
    """Extract facts from Bill of Materials (CSV/Excel)"""

    def extract(self, evidence_id: str, store: EvidenceStore) -> ExtractedEvidence:
        warnings: List[str] = []
        facts: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}
        metadata: Dict[str, Any] = {
            "row_count": 0,
            "column_count": 0,
            "parser": "unknown",
            "content_type": "unknown",
        }

        record = store.get(evidence_id)
        if record is None:
            warnings.append("evidence_not_found")
            return ExtractedEvidence(
                facts=_sorted_dict(facts),
                provenance=_sorted_dict(provenance),
                warnings=warnings,
                confidence=0.0,
                extraction_metadata=_sorted_dict(metadata),
            )

        metadata["content_type"] = record.content_type
        if record.content_type.lower() not in _SUPPORTED_CONTENT_TYPES:
            warnings.append("unsupported_content_type")
            return ExtractedEvidence(
                facts=_sorted_dict(facts),
                provenance=_sorted_dict(provenance),
                warnings=warnings,
                confidence=0.0,
                extraction_metadata=_sorted_dict(metadata),
            )

        blob_path = store.data_dir / evidence_id
        if not blob_path.exists():
            warnings.append("evidence_blob_missing")
            return ExtractedEvidence(
                facts=_sorted_dict(facts),
                provenance=_sorted_dict(provenance),
                warnings=warnings,
                confidence=0.0,
                extraction_metadata=_sorted_dict(metadata),
            )

        try:
            blob = blob_path.read_bytes()
        except Exception:
            warnings.append("evidence_blob_unreadable")
            return ExtractedEvidence(
                facts=_sorted_dict(facts),
                provenance=_sorted_dict(provenance),
                warnings=warnings,
                confidence=0.0,
                extraction_metadata=_sorted_dict(metadata),
            )

        rows, columns, parser = self._load_rows(blob, record.content_type, warnings)
        metadata["parser"] = parser
        metadata["row_count"] = len(rows)
        metadata["column_count"] = len(columns)

        normalized_rows, normalized_columns = self._normalize_rows(rows, columns)

        material_present = "material" in normalized_columns
        quantity_present = "quantity" in normalized_columns
        origin_present = "origin" in normalized_columns
        total_value_present = "total_value" in normalized_columns
        unit_price_present = "unit_price" in normalized_columns

        if not material_present:
            warnings.append("missing_column:material")
        if not origin_present:
            warnings.append("missing_column:origin")
        if not (total_value_present or unit_price_present):
            warnings.append("missing_column:value")

        material_weights: Dict[str, float] = {}
        textile_weights: Dict[str, float] = {}
        copper_conductor = False
        steel_component = False

        for row in normalized_rows:
            material_raw = row.get("material")
            if material_raw is None:
                continue
            material = _normalize_material(str(material_raw))
            if not material:
                continue
            if "copper" in material:
                copper_conductor = True
            if "steel" in material or "stainless" in material:
                steel_component = True

            weight = 1.0
            if quantity_present:
                qty_value = _to_float(row.get("quantity"))
                weight = qty_value if qty_value is not None else 0.0
            material_weights[material] = material_weights.get(material, 0.0) + weight

            for textile in _TEXTILE_MATERIALS:
                if textile in material:
                    textile_weights[textile] = textile_weights.get(textile, 0.0) + weight
                    break

        if material_weights:
            total_weight = sum(material_weights.values())
            if total_weight > 0:
                facts["_component_materials"] = _sorted_dict(
                    {key: (value / total_weight) * 100 for key, value in material_weights.items()}
                )
                provenance["_component_materials"] = f"bom:material_column:row_{len(normalized_rows)}"

        if textile_weights:
            total_textile = sum(textile_weights.values())
            if total_textile > 0:
                facts["textile_content_pct"] = _sorted_dict(
                    {key: (value / total_textile) * 100 for key, value in textile_weights.items()}
                )
                provenance["textile_content_pct"] = f"bom:material_column:row_{len(normalized_rows)}"

        if material_present:
            facts["copper_conductor"] = copper_conductor
            facts["steel_component"] = steel_component
            provenance["copper_conductor"] = f"bom:material_column:row_{len(normalized_rows)}"
            provenance["steel_component"] = f"bom:material_column:row_{len(normalized_rows)}"

        origins: Dict[str, int] = {}
        origin_display: Dict[str, str] = {}
        for row in normalized_rows:
            origin_raw = row.get("origin")
            if origin_raw is None:
                continue
            origin = _normalize_origin(str(origin_raw))
            if not origin:
                continue
            origin_key = origin.lower()
            origins[origin_key] = origins.get(origin_key, 0) + 1
            origin_display.setdefault(origin_key, origin)

        if origins:
            max_count = max(origins.values())
            candidate_keys = [key for key, count in origins.items() if count == max_count]
            primary_origin_key = sorted(candidate_keys)[0]
            facts["primary_origin"] = origin_display[primary_origin_key]
            facts["multi_origin"] = len(origins) > 1
            facts["origin_countries"] = [
                origin_display[key] for key in sorted(origin_display.keys())
            ]
            provenance["primary_origin"] = "bom:origin_column:aggregated"
            provenance["multi_origin"] = "bom:origin_column:aggregated"
            provenance["origin_countries"] = "bom:origin_column:aggregated"

        total_component_value = 0.0
        value_by_material: Dict[str, float] = {}
        value_detected = False

        for row in normalized_rows:
            value = None
            if total_value_present:
                value = _to_float(row.get("total_value"))
            if value is None and unit_price_present and quantity_present:
                unit_price = _to_float(row.get("unit_price"))
                quantity = _to_float(row.get("quantity"))
                if unit_price is not None and quantity is not None:
                    value = unit_price * quantity
            if value is None and unit_price_present:
                value = _to_float(row.get("unit_price"))

            if value is None:
                continue
            value_detected = True
            total_component_value += value

            material_raw = row.get("material")
            if material_raw is None:
                continue
            material = _normalize_material(str(material_raw))
            if not material:
                continue
            value_by_material[material] = value_by_material.get(material, 0.0) + value

        if total_value_present or unit_price_present:
            facts["total_component_value"] = total_component_value
            provenance["total_component_value"] = "bom:value_calculation:sum"

        if value_by_material:
            facts["value_by_material"] = _sorted_dict(value_by_material)
            provenance["value_by_material"] = "bom:value_calculation:sum"

        if not value_detected and (total_value_present or unit_price_present):
            warnings.append("value_parse_failed")

        confidence = 0.7 if facts else 0.0
        if warnings:
            confidence = min(confidence, 0.4)

        return ExtractedEvidence(
            facts=_sorted_dict(facts),
            provenance=_sorted_dict(provenance),
            warnings=warnings,
            confidence=confidence,
            extraction_metadata=_sorted_dict(metadata),
        )

    def _normalize_rows(
        self, rows: Iterable[Dict[str, Any]], columns: Iterable[str]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        normalized_columns: List[str] = []
        column_map: Dict[str, str] = {}

        for column in columns:
            normalized = self._resolve_column(column)
            if normalized and normalized not in column_map.values():
                column_map[column] = normalized
                normalized_columns.append(normalized)

        normalized_rows: List[Dict[str, Any]] = []
        for row in rows:
            normalized_row: Dict[str, Any] = {}
            for column, value in row.items():
                if column not in column_map:
                    continue
                normalized_key = column_map[column]
                normalized_row[normalized_key] = value
            normalized_rows.append(normalized_row)

        return normalized_rows, normalized_columns

    def _resolve_column(self, column: str) -> Optional[str]:
        normalized = _normalize_column(str(column))
        if not normalized:
            return None

        best_match: Optional[Tuple[int, str]] = None
        for canonical, aliases in COLUMN_MAP.items():
            for alias in aliases:
                alias_norm = _normalize_column(alias)
                if normalized == alias_norm:
                    return canonical
                if alias_norm and alias_norm in normalized:
                    candidate = (len(alias_norm), canonical)
                    if best_match is None or candidate > best_match:
                        best_match = candidate

        return best_match[1] if best_match else None

    def _load_rows(
        self, blob: bytes, content_type: str, warnings: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str], str]:
        if _module_available("pandas"):
            return self._load_rows_pandas(blob, content_type, warnings)
        return self._load_rows_fallback(blob, content_type, warnings)

    def _load_rows_pandas(
        self, blob: bytes, content_type: str, warnings: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str], str]:
        import pandas as pd

        buffer = io.BytesIO(blob)
        try:
            if content_type == "text/csv":
                df = pd.read_csv(buffer)
            elif content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                if not _module_available("openpyxl"):
                    warnings.append("xlsx_support_unavailable")
                    return [], [], "pandas"
                df = pd.read_excel(buffer, engine="openpyxl")
            elif content_type == "application/vnd.ms-excel":
                if not _module_available("xlrd"):
                    warnings.append("xls_support_unavailable")
                    return [], [], "pandas"
                df = pd.read_excel(buffer, engine="xlrd")
            else:
                warnings.append("unsupported_content_type")
                return [], [], "pandas"
        except Exception:
            warnings.append("bom_parse_failed")
            return [], [], "pandas"

        df = df.where(pd.notnull(df), None)
        columns = [str(column) for column in df.columns]
        records = df.to_dict(orient="records")
        return records, columns, "pandas"

    def _load_rows_fallback(
        self, blob: bytes, content_type: str, warnings: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[str], str]:
        if content_type == "text/csv":
            text = blob.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            columns = reader.fieldnames or []
            rows = [row for row in reader]
            return rows, columns, "csv"

        if content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            if not _module_available("openpyxl"):
                warnings.append("xlsx_support_unavailable")
                return [], [], "openpyxl"
            openpyxl = importlib.import_module("openpyxl")
            workbook = openpyxl.load_workbook(io.BytesIO(blob), read_only=True)
            sheet = workbook.active
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                return [], [], "openpyxl"
            headers = [str(header) if header is not None else "" for header in rows[0]]
            records: List[Dict[str, Any]] = []
            for row in rows[1:]:
                record = {headers[idx]: row[idx] for idx in range(len(headers))}
                records.append(record)
            return records, headers, "openpyxl"

        if content_type == "application/vnd.ms-excel":
            warnings.append("xls_support_unavailable")
            return [], [], "fallback"

        warnings.append("unsupported_content_type")
        return [], [], "fallback"
