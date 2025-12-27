from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from potatobacon.tariff.evidence.bom_extractor import BOMExtractor
from potatobacon.tariff.evidence_store import EvidenceStore


def _make_store(tmp_path: Path) -> EvidenceStore:
    index_path = tmp_path / "evidence_index.jsonl"
    data_dir = tmp_path / "blobs"
    return EvidenceStore(index_path=index_path, data_dir=data_dir)


def test_bom_material_extraction(tmp_path: Path):
    """BOM with materials → extract material composition"""
    csv_content = """component,material,quantity
Wire,Copper,10
Insulation,PVC,5
Connector,Plastic,2
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["copper_conductor"] is True

    materials = result.facts["_component_materials"]
    assert "copper" in materials
    assert "pvc" in materials
    assert abs(materials["copper"] - 58.8) < 0.1
    assert list(materials.keys()) == sorted(materials.keys())


def test_bom_origin_extraction(tmp_path: Path):
    """BOM with origins → extract primary/multi/list"""
    csv_content = """component,origin,value
Part A,China,100
Part B,China,50
Part C,Vietnam,30
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["primary_origin"] == "China"
    assert result.facts["multi_origin"] is True
    assert result.facts["origin_countries"] == ["China", "Vietnam"]


def test_bom_value_calculation(tmp_path: Path):
    """BOM with unit_price * quantity → total value"""
    csv_content = """component,unit_price,quantity
Part A,10.50,100
Part B,5.25,200
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["total_component_value"] == 2100.0


def test_bom_column_normalization(tmp_path: Path):
    """BOM with non-standard columns → normalize successfully"""
    csv_content = """Part Desc,Mat,Qty
Widget,Steel,5
Bracket,Aluminum,3
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["steel_component"] is True
    assert "_component_materials" in result.facts


def test_bom_textile_extraction(tmp_path: Path):
    """BOM with textile materials → extract fiber content"""
    csv_content = """material,quantity
Cotton,60
Polyester,40
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert "textile_content_pct" in result.facts
    assert result.facts["textile_content_pct"]["cotton"] == 60.0
    assert result.facts["textile_content_pct"]["polyester"] == 40.0


def test_bom_value_by_material(tmp_path: Path):
    """BOM with material+value → group value by material"""
    csv_content = """material,total_value
Copper,500
Steel,300
Copper,200
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    value_by_mat = result.facts["value_by_material"]
    assert value_by_mat["copper"] == 700.0
    assert value_by_mat["steel"] == 300.0
    assert list(value_by_mat.keys()) == sorted(value_by_mat.keys())


def test_bom_missing_columns(tmp_path: Path):
    """BOM with minimal columns → extract what's possible, warn"""
    csv_content = """description
Some part
Another part
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert result.warnings
    assert result.confidence < 0.5


def test_bom_xlsx_support(tmp_path: Path):
    """XLSX file → extract successfully"""
    openpyxl = pytest.importorskip("openpyxl")
    Workbook = openpyxl.Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["material", "quantity"])
    ws.append(["Copper", 10])
    ws.append(["Steel", 5])

    xlsx_bytes = io.BytesIO()
    wb.save(xlsx_bytes)
    xlsx_bytes.seek(0)

    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(
        xlsx_bytes.read(),
        filename="bom.xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    result = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["copper_conductor"] is True
    assert result.facts["steel_component"] is True


def test_bom_determinism(tmp_path: Path):
    """Same BOM → identical extraction"""
    csv_content = """material,quantity,origin
Copper,10,China
Steel,5,Vietnam
"""
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(csv_content.encode(), filename="bom.csv", content_type="text/csv")

    result1 = BOMExtractor().extract(record.evidence_id, evidence_store)
    result2 = BOMExtractor().extract(record.evidence_id, evidence_store)

    assert json.dumps(result1.facts, sort_keys=True) == json.dumps(result2.facts, sort_keys=True)
