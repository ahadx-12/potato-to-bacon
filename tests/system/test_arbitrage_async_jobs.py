import time

from tests.system.test_arbitrage_sync import _assert_arbitrage_dossier


def test_arbitrage_async_job_transparency(authed_client, bulk_manifest):
    payload = {
        "manifest_hash": "latest",
        "request": {
            "jurisdictions": ["US", "IE", "KY"],
            "domain": "tax",
            "objective": "MAXIMIZE_NET_AFTER_TAX",
            "constraints": {"entity_type": ["corp"], "risk_tolerance": "medium"},
            "seed": 424242,
        },
    }

    submitted = authed_client.post("/api/law/arbitrage/hunt/job", json=payload)
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["job_id"]
    assert job_id

    status = None
    for _ in range(20):
        status = authed_client.get(f"/api/law/jobs/{job_id}")
        assert status.status_code == 200, status.text
        body = status.json()
        if body.get("status") == "completed":
            break
        time.sleep(0.1)

    assert status is not None
    result_body = status.json()
    assert result_body.get("status") == "completed"
    dossier = result_body.get("result")
    assert dossier

    _assert_arbitrage_dossier(dossier, payload["request"]["jurisdictions"], payload["request"]["seed"])
