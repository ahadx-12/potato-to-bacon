import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("system_client")
def test_analysis_session_refine_flow(system_client: TestClient):
    sku_id = "SKU-ANALYSIS-FLOW"
    sku_payload = {
        "sku_id": sku_id,
        "description": "Electronics enclosure with harness and PCB slot",
        "declared_value_per_unit": 120.0,
    }
    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text

    session_resp = system_client.post(f"/api/tariff/skus/{sku_id}/sessions", json={})
    assert session_resp.status_code == 200, session_resp.text
    session = session_resp.json()
    assert session["sku_id"] == sku_id

    upload = system_client.post(
        "/api/tariff/evidence/upload",
        files={"file": ("evidence.json", b'{"doc": true}', "application/json")},
    )
    assert upload.status_code == 200, upload.text
    evidence_id = upload.json()["evidence_id"]

    refine_payload = {
        "attached_evidence_ids": [evidence_id],
        "fact_overrides": {
            "origin_country_US": {
                "value": True,
                "source": "certificate",
                "confidence": 0.95,
                "evidence_ids": [evidence_id],
            }
        },
        "evidence_requested": True,
        "optimize": True,
    }
    refine_resp = system_client.post(f"/api/tariff/sessions/{session['session_id']}/refine", json=refine_payload)
    assert refine_resp.status_code == 200, refine_resp.text
    body = refine_resp.json()
    dossier = body["dossier"]
    updated_session = body["session"]

    assert evidence_id in updated_session["attached_evidence_ids"]
    assert dossier["analysis_session_id"] == updated_session["session_id"]
    assert dossier["attached_evidence_ids"] == [evidence_id]
    assert "origin_country" not in dossier["questions"]["missing_facts"]
    assert dossier["fact_overrides"]

    proof_id = dossier["proof_id"]
    proof_evidence = system_client.get(f"/v1/proofs/{proof_id}/evidence")
    assert proof_evidence.status_code == 200, proof_evidence.text
    proof_body = proof_evidence.json()
    assert proof_body["analysis_session"]["session_id"] == updated_session["session_id"]
    assert evidence_id in proof_body["analysis_session"]["attached_evidence_ids"]
    assert proof_body["sku_metadata"].get("description_hash")
    assert "description" not in (proof_body["sku_metadata"] or {})


@pytest.mark.usefixtures("system_client")
def test_session_refine_persists_evidence_and_is_deterministic(system_client: TestClient):
    sku_id = "SKU-SESSION-DETERMINISTIC"
    sku_payload = {
        "sku_id": sku_id,
        "description": "Textile sneaker with rubber outsole",
        "declared_value_per_unit": 33.0,
    }
    create_resp = system_client.post("/api/tariff/skus", json=sku_payload)
    assert create_resp.status_code == 200, create_resp.text

    session_resp = system_client.post(f"/api/tariff/skus/{sku_id}/sessions", json={})
    assert session_resp.status_code == 200, session_resp.text
    session_id = session_resp.json()["session_id"]

    upload = system_client.post(
        "/api/tariff/evidence/upload",
        files={"file": ("evidence.json", b'{"doc": true}', "application/json")},
    )
    assert upload.status_code == 200, upload.text
    evidence_id = upload.json()["evidence_id"]

    refine_payload = {"attached_evidence_ids": [evidence_id], "fact_overrides": {}, "evidence_requested": True}
    first = system_client.post(f"/api/tariff/sessions/{session_id}/refine", json=refine_payload)
    assert first.status_code == 200, first.text
    first_body = first.json()

    second = system_client.post(f"/api/tariff/sessions/{session_id}/refine", json=refine_payload)
    assert second.status_code == 200, second.text
    second_body = second.json()

    first_dossier = first_body["dossier"]
    second_dossier = second_body["dossier"]
    assert first_dossier["attached_evidence_ids"] == [evidence_id]
    assert second_dossier["attached_evidence_ids"] == [evidence_id]
    assert first_dossier["questions"]["missing_facts"] == second_dossier["questions"]["missing_facts"]
    assert first_dossier["baseline"]["candidates"]
    assert second_dossier["baseline"]["candidates"]
    assert first_dossier["baseline"]["candidates"][0]["candidate_id"] == second_dossier["baseline"]["candidates"][0]["candidate_id"]

    proof_id = first_dossier["proof_id"]
    proof_evidence = system_client.get(f"/v1/proofs/{proof_id}/evidence")
    assert proof_evidence.status_code == 200, proof_evidence.text
    proof_body = proof_evidence.json()
    assert proof_body["analysis_session"]["session_id"] == session_id
    assert evidence_id in proof_body["analysis_session"]["attached_evidence_ids"]


def test_tariff_session_endpoints_require_auth(make_client):
    with make_client(headers=None) as (client, _env):
        resp = client.post(
            "/api/tariff/evidence/upload",
            files={"file": ("demo.pdf", b"1234", "application/pdf")},
        )
        assert resp.status_code == 401

        session_resp = client.post("/api/tariff/skus/UNKNOWN/sessions", json={})
        assert session_resp.status_code == 401
