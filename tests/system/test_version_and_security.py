import time
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


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


def test_version_and_api_key_enforcement(make_client):
    payload = conflict_payload()
    with make_client() as (client, _):
        version = client.get("/v1/version")
        assert version.status_code == 200
        body = version.json()
        assert "engine_version" in body
        assert "manifest_hash" in body

        unauthorized = client.post("/v1/law/analyze", json=payload)
        assert unauthorized.status_code == 401

    with make_client(headers={"X-API-Key": "dev-key"}) as (client, _):
        authorized = client.post("/v1/law/analyze", json=payload)
        assert authorized.status_code == 200


def test_rate_limiting_with_custom_window(make_client, monkeypatch):
    payload = conflict_payload()
    with make_client(
        extra_env={"CALE_RATE_WINDOW_SEC": "2", "CALE_RATE_LIMIT_PER_MINUTE": "2"},
        headers={"X-API-Key": "dev-key"},
    ) as (client, _):
        from potatobacon.api import security as security_module

        security_module.set_rate_limit(2)
        security_module.rate_limiter.window_seconds = 2
        security_module.rate_limiter.reset()
        real_time = time.time
        start = real_time()
        monkeypatch.setattr(security_module.time, "time", lambda: start)

        first = client.post("/v1/law/analyze", json=payload)
        second = client.post("/v1/law/analyze", json=payload)
        assert first.status_code == 200
        assert second.status_code == 200

        burst = client.post("/v1/law/analyze", json=payload)
        assert burst.status_code == 429

        different_key = client.post(
            "/v1/law/analyze", json=payload, headers={"X-API-Key": "alt-key"}
        )
        assert different_key.status_code == 200

        monkeypatch.setattr(security_module.time, "time", real_time)
        time.sleep(2)
        after_reset = client.post("/v1/law/analyze", json=payload)
        assert after_reset.status_code == 200
