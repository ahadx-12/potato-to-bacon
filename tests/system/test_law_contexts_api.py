import pytest

from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID


@pytest.mark.usefixtures("system_client")
def test_list_law_contexts(system_client):
    response = system_client.get("/v1/law-contexts")
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["default_context"] == DEFAULT_CONTEXT_ID
    assert payload.get("contexts")
    assert any(ctx["context_id"] == DEFAULT_CONTEXT_ID for ctx in payload["contexts"])


@pytest.mark.usefixtures("system_client")
def test_get_default_context(system_client):
    response = system_client.get(f"/v1/law-contexts/{DEFAULT_CONTEXT_ID}")
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload.get("manifest_hash")
    assert payload.get("atoms_count", 0) > 0


@pytest.mark.usefixtures("system_client")
def test_invalid_law_context_errors(system_client):
    suggest_request = {
        "description": "Canvas sneaker with rubber sole",
        "law_context": "UNKNOWN_CONTEXT",
    }

    suggest_response = system_client.post("/api/tariff/suggest", json=suggest_request)
    assert suggest_response.status_code == 400, suggest_response.text
    suggest_detail = suggest_response.json()["detail"]
    assert "available_contexts" in suggest_detail

    base_facts = {
        "upper_material_textile": True,
        "outer_sole_material_rubber_or_plastics": True,
        "surface_contact_rubber_gt_50": True,
        "surface_contact_textile_gt_50": False,
        "felt_covering_gt_50": False,
    }
    optimize_request = {
        "scenario": base_facts,
        "candidate_mutations": {"felt_covering_gt_50": [False, True]},
        "law_context": "UNKNOWN_CONTEXT",
    }

    optimize_response = system_client.post("/api/tariff/optimize", json=optimize_request)
    assert optimize_response.status_code == 400, optimize_response.text
    optimize_detail = optimize_response.json()["detail"]
    assert "available_contexts" in optimize_detail
