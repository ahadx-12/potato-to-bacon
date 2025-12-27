from __future__ import annotations

import re
from typing import Any, Dict, Optional

from potatobacon.tariff.evidence.pdf_extractor import PDFExtractor, _sorted_dict
from potatobacon.tariff.evidence.types import ExtractedEvidence
from potatobacon.tariff.evidence_store import EvidenceStore


class CertificateExtractor(PDFExtractor):
    def _detect_cert_type(self, text_sample: str) -> Optional[str]:
        if re.search(r"\bUL\b", text_sample, flags=re.IGNORECASE) and (
            re.search(r"UL File Number", text_sample, flags=re.IGNORECASE)
            or re.search(r"UL Listed", text_sample, flags=re.IGNORECASE)
        ):
            return "UL"
        if re.search(r"ISO\s*(9001|14001)", text_sample, flags=re.IGNORECASE):
            return "ISO"
        if re.search(r"Certificate of Origin", text_sample, flags=re.IGNORECASE):
            return "COO"
        if re.search(r"\bconsignee\b", text_sample, flags=re.IGNORECASE) and re.search(
            r"\bexporter\b", text_sample, flags=re.IGNORECASE
        ):
            return "COO"
        return None

    def extract(self, evidence_id: str, store: EvidenceStore) -> ExtractedEvidence:
        base = super().extract(evidence_id, store)

        text_sample = ""
        warnings = list(base.warnings)
        blob = self._load_blob(evidence_id, store, warnings)
        if blob is not None:
            text, _, _ = self._extract_text(blob, warnings)
            text_sample = text[:800]

        cert_type = self._detect_cert_type(text_sample)
        facts: Dict[str, Any] = dict(base.facts)
        provenance = dict(base.provenance)
        metadata = dict(base.extraction_metadata)
        if cert_type:
            metadata["cert_type"] = cert_type

        if cert_type == "UL":
            if "ul_listed" not in facts and re.search(r"UL Listed", text_sample, flags=re.IGNORECASE):
                facts["ul_listed"] = True
                provenance["ul_listed"] = "pdf:text:pattern:UL Listed"
            category_match = re.search(r"Product Category[:\s]+([A-Za-z\s]+)", text_sample, flags=re.IGNORECASE)
            if category_match:
                facts.setdefault("ul_product_category", category_match.group(1).strip())
                provenance.setdefault(
                    "ul_product_category", f"pdf:text:pattern:{category_match.group(0).strip()}"
                )

        if cert_type == "ISO":
            iso_match = re.search(r"ISO\s*(9001|14001)", text_sample, flags=re.IGNORECASE)
            if iso_match:
                facts.setdefault("iso_standard", f"ISO {iso_match.group(1)}")
                provenance.setdefault("iso_standard", f"pdf:text:pattern:{iso_match.group(0).strip()}")
            body_match = re.search(r"Certification Body[:\s]+([A-Za-z\s]+)", text_sample, flags=re.IGNORECASE)
            if body_match:
                facts.setdefault("certification_body", body_match.group(1).strip())
                provenance.setdefault(
                    "certification_body", f"pdf:text:pattern:{body_match.group(0).strip()}"
                )

        if cert_type == "COO":
            exporter_match = re.search(r"Exporter[:\s]+([A-Za-z0-9\s]+)", text_sample, flags=re.IGNORECASE)
            if exporter_match:
                facts.setdefault("exporter", exporter_match.group(1).strip())
                provenance.setdefault("exporter", f"pdf:text:pattern:{exporter_match.group(0).strip()}")
            consignee_match = re.search(r"Consignee[:\s]+([A-Za-z0-9\s]+)", text_sample, flags=re.IGNORECASE)
            if consignee_match:
                facts.setdefault("consignee", consignee_match.group(1).strip())
                provenance.setdefault("consignee", f"pdf:text:pattern:{consignee_match.group(0).strip()}")
            hs_match = re.search(r"HS Code[:\s]+([0-9\.]+)", text_sample, flags=re.IGNORECASE)
            if hs_match:
                facts.setdefault("hs_code", hs_match.group(1).strip())
                provenance.setdefault("hs_code", f"pdf:text:pattern:{hs_match.group(0).strip()}")

        if "textile_content_pct" in facts and isinstance(facts["textile_content_pct"], dict):
            facts["textile_content_pct"] = _sorted_dict(facts["textile_content_pct"])

        confidence = base.confidence
        return ExtractedEvidence(
            facts=_sorted_dict(facts),
            provenance=_sorted_dict(provenance),
            warnings=warnings,
            confidence=confidence,
            extraction_metadata=_sorted_dict(metadata),
        )
