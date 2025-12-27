from __future__ import annotations

import importlib
import importlib.util
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from potatobacon.tariff.evidence.types import ExtractedEvidence
from potatobacon.tariff.evidence_store import EvidenceStore


_TEXTILE_MATERIALS = [
    "cotton",
    "polyester",
    "nylon",
    "wool",
    "silk",
    "rayon",
    "spandex",
    "elastane",
]

_ORIGIN_PATTERNS = [
    r"Country of Origin[:\s]+([A-Za-z\s]+)",
    r"Made in[:\s]+([A-Za-z\s]+)",
    r"Origin[:\s]+([A-Za-z\s]+)",
]

_DIMENSION_UNITS = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
}

_WEIGHT_UNITS = {
    "g": 0.001,
    "kg": 1.0,
}

_SOURCE_CONFIDENCE = {
    "table": 0.9,
    "text": 0.8,
    "ocr": 0.6,
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sorted_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: data[key] for key in sorted(data.keys())}


def _normalize_space(value: str) -> str:
    cleaned = " ".join(value.split())
    return cleaned.strip()


def _normalize_country(value: str) -> str:
    return _normalize_space(value)


def _textile_regex() -> re.Pattern[str]:
    materials = "|".join(_TEXTILE_MATERIALS)
    return re.compile(rf"\b({materials})\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def _dimension_regex() -> re.Pattern[str]:
    return re.compile(r"\b(length|width|thickness)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", re.IGNORECASE)


def _weight_regex() -> re.Pattern[str]:
    return re.compile(r"\bweight\b\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", re.IGNORECASE)


class PDFExtractor:
    def extract(self, evidence_id: str, store: EvidenceStore) -> ExtractedEvidence:
        warnings: List[str] = []
        facts: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}
        fact_confidence: Dict[str, float] = {}
        textile_confidence: Dict[str, float] = {}
        metadata: Dict[str, Any] = {
            "method": "none",
            "page_count": 0,
            "text_length": 0,
            "used_ocr": False,
            "table_rows": 0,
        }

        blob = self._load_blob(evidence_id, store, warnings)
        if blob is None:
            warnings.append("extraction_quality: low")
            return ExtractedEvidence(
                facts=_sorted_dict(facts),
                provenance=_sorted_dict(provenance),
                warnings=warnings,
                confidence=0.0,
                extraction_metadata=_sorted_dict(metadata),
            )

        text, method, page_count = self._extract_text(blob, warnings)
        metadata["method"] = method
        metadata["page_count"] = page_count
        metadata["text_length"] = len(text)

        text_facts, text_provenance = self._extract_facts_from_text(text, "pdf:text:pattern")
        self._merge_facts(
            facts,
            provenance,
            fact_confidence,
            textile_confidence,
            text_facts,
            text_provenance,
            _SOURCE_CONFIDENCE["text"],
        )

        table_rows, table_facts, table_provenance = self._extract_table_facts(blob, warnings)
        metadata["table_rows"] = table_rows
        self._merge_facts(
            facts,
            provenance,
            fact_confidence,
            textile_confidence,
            table_facts,
            table_provenance,
            _SOURCE_CONFIDENCE["table"],
        )

        if len(text) < 50:
            ocr_text = self._extract_ocr_text(blob, warnings)
            if ocr_text:
                metadata["used_ocr"] = True
                ocr_facts, ocr_provenance = self._extract_facts_from_text(ocr_text, "pdf:ocr")
                self._merge_facts(
                    facts,
                    provenance,
                    fact_confidence,
                    textile_confidence,
                    ocr_facts,
                    ocr_provenance,
                    _SOURCE_CONFIDENCE["ocr"],
                )

        if not facts:
            warnings.append("extraction_quality: low")

        if "textile_content_pct" in facts and isinstance(facts["textile_content_pct"], dict):
            facts["textile_content_pct"] = _sorted_dict(facts["textile_content_pct"])

        confidence = max(fact_confidence.values(), default=0.0)
        return ExtractedEvidence(
            facts=_sorted_dict(facts),
            provenance=_sorted_dict(provenance),
            warnings=warnings,
            confidence=confidence,
            extraction_metadata=_sorted_dict(metadata),
        )

    def _load_blob(self, evidence_id: str, store: EvidenceStore, warnings: List[str]) -> Optional[bytes]:
        record = store.get(evidence_id)
        if record is None:
            warnings.append("evidence_not_found")
            return None
        blob_path = store.data_dir / evidence_id
        if not blob_path.exists():
            warnings.append("evidence_blob_missing")
            return None
        try:
            return blob_path.read_bytes()
        except Exception:
            warnings.append("evidence_blob_unreadable")
            return None

    def _extract_text(self, blob: bytes, warnings: List[str]) -> Tuple[str, str, int]:
        if _module_available("fitz"):
            fitz = importlib.import_module("fitz")
            try:
                doc = fitz.open(stream=blob, filetype="pdf")
                try:
                    pages = [page.get_text() or "" for page in doc]
                finally:
                    doc.close()
                return "\n".join(pages), "pymupdf", len(pages)
            except Exception:
                warnings.append("text_extraction_failed:pymupdf")

        try:
            with pdfplumber.open(io.BytesIO(blob)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages), "pdfplumber", len(pages)
        except Exception:
            warnings.append("text_extraction_failed:pdfplumber")
            return "", "none", 0

    def _extract_table_facts(
        self, blob: bytes, warnings: List[str]
    ) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        facts: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}
        fact_confidence: Dict[str, float] = {}
        textile_confidence: Dict[str, float] = {}
        row_index = 0
        try:
            with pdfplumber.open(io.BytesIO(blob)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables() or []
                    for table in tables:
                        for row in table:
                            row_index += 1
                            row_text = " ".join(cell for cell in row if cell) if row else ""
                            if not row_text.strip():
                                continue
                            row_facts, row_provenance = self._extract_facts_from_text(
                                row_text, f"pdf:table:row_{row_index}"
                            )
                            self._merge_facts(
                                facts,
                                provenance,
                                fact_confidence,
                                textile_confidence,
                                row_facts,
                                row_provenance,
                                _SOURCE_CONFIDENCE["table"],
                            )
        except Exception:
            warnings.append("table_extraction_failed")
        return row_index, facts, provenance

    def _extract_ocr_text(self, blob: bytes, warnings: List[str]) -> str:
        if not (_module_available("pytesseract") and _module_available("PIL")):
            warnings.append("ocr_unavailable")
            return ""
        if not _module_available("fitz"):
            warnings.append("ocr_render_unavailable")
            return ""

        pytesseract = importlib.import_module("pytesseract")
        fitz = importlib.import_module("fitz")
        image_module = importlib.import_module("PIL.Image")

        text_chunks: List[str] = []
        try:
            doc = fitz.open(stream=blob, filetype="pdf")
            try:
                for page in doc:
                    pix = page.get_pixmap()
                    image_bytes = pix.tobytes("png")
                    with image_module.open(io.BytesIO(image_bytes)) as image:
                        text_chunks.append(pytesseract.image_to_string(image) or "")
            finally:
                doc.close()
        except Exception:
            warnings.append("ocr_failed")
            return ""

        return "\n".join(text_chunks)

    def _extract_facts_from_text(self, text: str, provenance_prefix: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
        facts: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}
        if not text:
            return facts, provenance

        for pattern in _ORIGIN_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                origin_raw = match.group(1)
                origin = _normalize_country(origin_raw.splitlines()[0])
                facts["origin_country"] = origin
                provenance["origin_country"] = f"{provenance_prefix}:{match.group(0).strip()}"
                break

        textile_matches = list(_textile_regex().finditer(text))
        if textile_matches:
            textile: Dict[str, float] = {}
            for match in textile_matches:
                material = match.group(1).lower()
                pct = float(match.group(2))
                textile[material] = pct
            facts["textile_content_pct"] = textile
            provenance["textile_content_pct"] = f"{provenance_prefix}:textile_content"

        lowered = text.lower()
        if "copper conductor" in lowered or "copper wire" in lowered:
            facts["copper_conductor"] = True
            provenance["copper_conductor"] = f"{provenance_prefix}:copper"
        if "stainless steel" in lowered or "steel" in lowered:
            facts["steel_component"] = True
            provenance["steel_component"] = f"{provenance_prefix}:steel"

        for match in _dimension_regex().finditer(text):
            label = match.group(1).lower()
            magnitude = float(match.group(2))
            unit = match.group(3).lower()
            if unit not in _DIMENSION_UNITS:
                continue
            mm_value = magnitude * _DIMENSION_UNITS[unit]
            key = f"{label}_mm"
            if key not in facts:
                facts[key] = mm_value
                provenance[key] = f"{provenance_prefix}:{match.group(0).strip()}"

        weight_match = _weight_regex().search(text)
        if weight_match:
            magnitude = float(weight_match.group(1))
            unit = weight_match.group(2).lower()
            if unit in _WEIGHT_UNITS:
                facts["weight_kg"] = magnitude * _WEIGHT_UNITS[unit]
                provenance["weight_kg"] = f"{provenance_prefix}:{weight_match.group(0).strip()}"

        if "ul listed" in lowered or "ul-listed" in lowered:
            facts["ul_listed"] = True
            provenance["ul_listed"] = f"{provenance_prefix}:ul_listed"
        if "ce marked" in lowered or "ce mark" in lowered or "ce-marked" in lowered:
            facts["ce_marked"] = True
            provenance["ce_marked"] = f"{provenance_prefix}:ce_marked"
        if "rohs" in lowered:
            facts["rohs_compliant"] = True
            provenance["rohs_compliant"] = f"{provenance_prefix}:rohs"

        return facts, provenance

    def _merge_facts(
        self,
        facts: Dict[str, Any],
        provenance: Dict[str, str],
        fact_confidence: Dict[str, float],
        textile_confidence: Dict[str, float],
        new_facts: Dict[str, Any],
        new_provenance: Dict[str, str],
        confidence: float,
    ) -> None:
        for key, value in new_facts.items():
            if key == "textile_content_pct" and isinstance(value, dict):
                existing = facts.get(key)
                if not isinstance(existing, dict):
                    existing = {}
                for material, pct in value.items():
                    current_conf = textile_confidence.get(material, -1.0)
                    if confidence > current_conf:
                        existing[material] = pct
                        textile_confidence[material] = confidence
                        fact_confidence[key] = max(fact_confidence.get(key, 0.0), confidence)
                        provenance[key] = new_provenance.get(key, "")
                facts[key] = existing
                continue

            current_conf = fact_confidence.get(key, -1.0)
            if confidence > current_conf:
                facts[key] = value
                fact_confidence[key] = confidence
                if key in new_provenance:
                    provenance[key] = new_provenance[key]
