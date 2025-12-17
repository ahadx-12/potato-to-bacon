
def test_suggest_returns_ad_cvd_flags(system_client):
    payload = {
        "sku_id": "BOLT-CN",
        "description": "Steel fastener bolt",
        "origin_country": "CN",
        "import_country": "US",
        "declared_value_per_unit": 50,
    }
    response = system_client.post("/api/tariff/suggest", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "OK"
    assert data["suggestions"], data

    top = data["suggestions"][0]
    assert any("AD/CVD" in reason for reason in top.get("risk_reasons", []))

    evidence_resp = system_client.get(f"/v1/proofs/{top['proof_id']}/evidence")
    assert evidence_resp.status_code == 200
    evidence_payload = evidence_resp.json()
    compiled_facts = evidence_payload.get("compiled_facts") or {}
    assert "origin_country_raw" in compiled_facts
