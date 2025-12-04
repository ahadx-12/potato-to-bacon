from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def conflict_payload() -> dict:
    aml_text = load_text("aml_privacy_pair.txt").strip()
    section_text = load_text("tax_ie_section_110.txt").strip()
    return {
        "rule1": {
            "text": f"MUST enforce_customer_due_diligence. {aml_text}",
            "jurisdiction": "EU",
            "statute": "AML",
            "section": "Privacy",
            "enactment_year": 2021,
        },
        "rule2": {
            "text": f"MAY structure_profit_participating_notes. {section_text}",
            "jurisdiction": "Ireland",
            "statute": "Section 110",
            "section": "SPV",
            "enactment_year": 1997,
        },
    }


def test_bulk_ingest_and_analysis_metrics(make_client):
    section_text = load_text("tax_ie_section_110.txt").strip()
    aml_text = load_text("aml_privacy_pair.txt").strip()
    sources = [
        {
            "id": "us_irc_61",
            "text": f"MUST report_all_income_annually. {load_text('tax_us_irc_61.txt').strip()}",
            "jurisdiction": "US",
            "statute": "IRC",
            "section": "61",
            "enactment_year": 2017,
        },
        {
            "id": "ie_section_110",
            "text": f"MAY apply_section_110_structures. {section_text}",
            "jurisdiction": "Ireland",
            "statute": "Finance Act",
            "section": "110",
            "enactment_year": 1997,
        },
        {
            "id": "cayman_zero_tax",
            "text": f"MAY exempt_foreign_income when entity_is_exempted. {load_text('cayman_corp_zero_tax.txt').strip()}",
            "jurisdiction": "Cayman Islands",
            "statute": "Companies Law",
            "section": "Tax",
            "enactment_year": 2020,
        },
        {
            "id": "aml_privacy",
            "text": f"MUST safeguard_personal_data. {aml_text}",
            "jurisdiction": "EU",
            "statute": "AML",
            "section": "Privacy",
            "enactment_year": 2021,
        },
    ]

    with make_client(headers={"X-API-Key": "dev-key"}) as (client, _):
        before = client.get("/v1/version").json().get("manifest_hash")
        response = client.post(
            "/v1/manifest/bulk_ingest",
            json={"domain": "tax", "sources": sources, "options": {"replace_existing": True}},
        )
        assert response.status_code == 200
        body = response.json()
        manifest_hash = body["manifest_hash"]
        assert manifest_hash
        if before:
            assert manifest_hash != before

        analyze = client.post("/v1/law/analyze", json=conflict_payload())
        assert analyze.status_code == 200
        report = analyze.json()
        assert report.get("manifest_hash") == manifest_hash

        samples = report["scenario_metrics"]["samples"]
        assert samples
        sample = samples[0]
        assert "entropy" in sample
        assert "kappa" in sample
        assert "value_estimate" in sample
        assert len(sample.get("active_rules", [])) > 0
