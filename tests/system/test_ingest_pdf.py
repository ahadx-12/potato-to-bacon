from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def ensure_pdf_exists() -> Path:
    pdf_path = FIXTURES / "test_statute.pdf"
    if pdf_path.exists():
        from potatobacon.law.pdf_ingest import extract_text_from_pdf

        extracted = extract_text_from_pdf(pdf_path.read_bytes())
        if "MUST" in extracted or "MAY" in extracted:
            return pdf_path

    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(72, 720, "MUST maintain_tax_records. Section 1. Filing obligations for tax residents.")
    c.drawString(
        72,
        700,
        "MAY claim_international_relief when qualifying_holding_company. Section 2 details.",
    )
    c.showPage()
    c.save()
    return pdf_path


def conflict_payload() -> dict:
    us_text = (FIXTURES / "tax_us_irc_61.txt").read_text().strip()
    cayman_text = (FIXTURES / "cayman_corp_zero_tax.txt").read_text().strip()
    return {
        "rule1": {
            "text": f"MUST report_all_income_annually. {us_text}",
            "jurisdiction": "US",
            "statute": "IRC",
            "section": "61",
            "enactment_year": 2017,
        },
        "rule2": {
            "text": f"MAY claim_offshore_tax_neutrality when non_resident. {cayman_text}",
            "jurisdiction": "Cayman Islands",
            "statute": "Companies Law",
            "section": "Tax",
            "enactment_year": 2020,
        },
    }


def test_pdf_ingest_updates_manifest(make_client):
    pdf_path = ensure_pdf_exists()
    files = {"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}

    with make_client(headers={"X-API-Key": "dev-key"}) as (client, _):
        before = client.get("/v1/version").json().get("manifest_hash")
        response = client.post(
            "/v1/manifest/ingest_pdf", data={"domain": "tax", "base_id": "statute"}, files=files
        )
        assert response.status_code == 200
        payload = response.json()
        manifest_hash = payload["manifest_hash"]
        assert manifest_hash
        if before:
            assert manifest_hash != before
        assert payload.get("sources_ingested")

        analysis = client.post("/v1/law/analyze", json=conflict_payload())
        assert analysis.status_code == 200
        report = analysis.json()
        assert report.get("manifest_hash") == manifest_hash
        sample = report["scenario_metrics"]["samples"][0]
        assert len(sample.get("active_rules", [])) > 0
