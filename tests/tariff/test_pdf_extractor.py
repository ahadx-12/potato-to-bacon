from __future__ import annotations

import json
from pathlib import Path

from potatobacon.tariff.evidence.certificate_extractor import CertificateExtractor
from potatobacon.tariff.evidence.pdf_extractor import PDFExtractor
from potatobacon.tariff.evidence_store import EvidenceStore
from tests.helpers.pdf_factory import create_test_pdf_with_table, create_test_pdf_with_text


def _make_store(tmp_path: Path) -> EvidenceStore:
    index_path = tmp_path / "evidence_index.jsonl"
    data_dir = tmp_path / "blobs"
    return EvidenceStore(index_path=index_path, data_dir=data_dir)


def test_pdf_origin_extraction(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("Country of Origin: Vietnam")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["origin_country"] == "Vietnam"
    assert "Country of Origin" in result.provenance["origin_country"]
    assert result.confidence > 0.7


def test_pdf_material_extraction(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("Material Composition:\nCotton: 60%\nPolyester: 40%")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["textile_content_pct"]["cotton"] == 60.0
    assert result.facts["textile_content_pct"]["polyester"] == 40.0
    assert list(result.facts["textile_content_pct"].keys()) == ["cotton", "polyester"]


def test_pdf_certification_extraction(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("This product is UL listed and RoHS compliant")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["ul_listed"] is True
    assert result.facts["rohs_compliant"] is True


def test_pdf_metal_component_flags(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("Includes copper conductor and stainless steel hardware")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["copper_conductor"] is True
    assert result.facts["steel_component"] is True


def test_pdf_table_extraction(tmp_path: Path):
    rows = [
        ["Property", "Value"],
        ["Country of Origin", "China"],
        ["Weight", "2.5 kg"],
    ]
    pdf_bytes = create_test_pdf_with_table(rows)
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["origin_country"] == "China"
    assert result.facts["weight_kg"] == 2.5
    assert "table" in result.provenance["origin_country"]


def test_pdf_dimension_extraction(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("Dimensions: Length: 100mm, Width: 5cm")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert result.facts["length_mm"] == 100.0
    assert result.facts["width_mm"] == 50.0


def test_pdf_empty_extraction(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("Lorem ipsum dolor sit amet")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert len(result.facts) == 0
    assert any("extraction_quality" in warning for warning in result.warnings)
    assert result.confidence < 0.3


def test_certificate_type_detection(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("UL LLC\nUL File Number: E123456\nUL Listed Product")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result = CertificateExtractor().extract(record.evidence_id, evidence_store)

    assert result.extraction_metadata.get("cert_type") == "UL"
    assert result.facts["ul_listed"] is True


def test_extraction_determinism(tmp_path: Path):
    pdf_bytes = create_test_pdf_with_text("Country of Origin: Vietnam\nCotton: 60%, Polyester: 40%")
    evidence_store = _make_store(tmp_path)
    record = evidence_store.save(pdf_bytes, filename="test.pdf", content_type="application/pdf")

    result1 = PDFExtractor().extract(record.evidence_id, evidence_store)
    result2 = PDFExtractor().extract(record.evidence_id, evidence_store)

    assert json.dumps(result1.facts, sort_keys=True) == json.dumps(result2.facts, sort_keys=True)
    assert result1.confidence == result2.confidence
