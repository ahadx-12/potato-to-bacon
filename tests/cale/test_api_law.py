import os

import os
from typing import Dict

os.environ["CALE_EMBED_BACKEND"] = "hash"

import pytest
from fastapi.testclient import TestClient

from potatobacon.api.app import app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CALE_API_KEYS", "test-key")
    monkeypatch.setenv("CALE_RATE_LIMIT_PER_MINUTE", "100")
    monkeypatch.delenv("CALE_DISABLE_STARTUP_INIT", raising=False)
    with TestClient(app, headers={"X-API-Key": "test-key"}) as test_client:
        yield test_client


def test_analyze_endpoint_returns_scores_and_components(client):
    request = {
        "rule1": {
            "text": "Organizations MUST collect personal data IF consent.",
            "jurisdiction": "CA.Federal",
            "statute": "PIPEDA",
            "section": "7(3)",
            "enactment_year": 2000,
        },
        "rule2": {
            "text": "Security agencies MUST NOT collect personal data IF emergency.",
            "jurisdiction": "CA.Federal",
            "statute": "ATA",
            "section": "83.28",
            "enactment_year": 2001,
        },
    }

    response = client.post("/v1/law/analyze", json=request)
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body["conflict_scores"].keys()) == {"textualist", "living", "pragmatic"}
    components = body["components"]
    for key in ("symbolic_conflict", "contextual_similarity", "authority", "temporal_drift"):
        assert key in components
    assert 0.0 <= body["variance"] <= 1.0


def _suggest_request() -> Dict[str, Dict[str, object]]:
    return {
        "rule1": {
            "text": "Organizations MUST collect personal data IF consent.",
            "jurisdiction": "Canada.Federal",
            "statute": "PIPEDA",
            "section": "7(3)",
            "enactment_year": 2000,
        },
        "rule2": {
            "text": "Security agencies MUST NOT collect personal data IF emergency.",
            "jurisdiction": "Canada.Federal",
            "statute": "Anti-Terrorism Act",
            "section": "83.28",
            "enactment_year": 2001,
        },
    }


def test_suggest_endpoint_schema(client):
    response = client.post("/v1/law/suggest_amendment", json=_suggest_request())
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body.keys()) == {
        "precedent_count",
        "candidates_considered",
        "suggestions",
        "best",
    }
    assert isinstance(body["suggestions"], list)
    assert len(body["suggestions"]) <= 3
    if body["suggestions"]:
        for item in body["suggestions"]:
            assert set(item.keys()) == {
                "condition",
                "justification",
                "estimated_ccs",
                "suggested_text",
            }
            assert set(item["justification"].keys()) >= {
                "frequency",
                "semantic_relevance",
                "impact",
                "composite_score",
            }


def test_analyze_plus_suggest_roundtrip(client):
    request = _suggest_request()
    analyze = client.post("/v1/law/analyze", json=request)
    assert analyze.status_code == 200, analyze.text
    conflict = analyze.json()["conflict_scores"]["pragmatic"]
    suggest = client.post("/v1/law/suggest_amendment", json=request)
    assert suggest.status_code == 200, suggest.text
    body = suggest.json()
    assert body["precedent_count"] >= 1
    assert len(body["suggestions"]) <= 3
    if body["best"] is not None:
        assert body["best"]["estimated_ccs"] <= conflict
