import pytest

from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID


@pytest.mark.usefixtures("system_client")
def test_electronics_dossier_uses_ingested_hts_slice(system_client):
    payload = {
        "description": "USB-C cable assembly with copper conductors, molded connectors, and insulation",
        "declared_value_per_unit": 18.0,
        "annual_volume": 5000,
        "law_context": DEFAULT_CONTEXT_ID,
    }

    response = system_client.post("/api/tariff/sku/dossier", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["law_context"] == DEFAULT_CONTEXT_ID
    assert body.get("tariff_manifest_hash")
    assert body.get("baseline_candidates")

    first_candidate = body["baseline_candidates"][0]
    assert first_candidate["provenance_chain"]
    provenance = first_candidate["provenance_chain"][0]
    assert provenance.get("citation")
    assert provenance["citation"].get("heading")
    metadata = provenance.get("metadata")
    assert metadata and metadata.get("hts_code")
