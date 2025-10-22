import os

os.environ["CALE_EMBED_BACKEND"] = "hash"

from fastapi.testclient import TestClient

from potatobacon.api.app import app

client = TestClient(app)


def test_analyze_endpoint_returns_scores_and_components():
    request = {
        "rule1": {
            "text": "Organizations MUST obtain consent before collecting personal data.",
            "jurisdiction": "CA.Federal",
            "statute": "PIPEDA",
            "section": "7(3)",
            "enactment_year": 2000,
        },
        "rule2": {
            "text": "Security agencies MAY access data without consent during emergencies.",
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
