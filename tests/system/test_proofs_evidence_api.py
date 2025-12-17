import pytest


@pytest.mark.usefixtures("system_client")
def test_proof_evidence_endpoint(system_client):
    suggest_payload = {
        "description": "Canvas sneaker with rubber sole",
        "include_fact_evidence": True,
    }

    suggest_response = system_client.post("/api/tariff/suggest", json=suggest_payload)
    assert suggest_response.status_code == 200, suggest_response.text
    suggestion = suggest_response.json()["suggestions"][0]
    proof_id = suggestion["proof_id"]

    evidence_response = system_client.get(f"/v1/proofs/{proof_id}/evidence")
    assert evidence_response.status_code == 200, evidence_response.text
    evidence_payload = evidence_response.json()
    assert evidence_payload.get("proof_id") == proof_id
    assert evidence_payload.get("fact_evidence") is not None
