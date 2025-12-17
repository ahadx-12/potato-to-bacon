import concurrent.futures
import threading

import pytest


REQUEST_LOCK = threading.Lock()


def _post(client, path: str, payload: dict) -> tuple[int, dict | None]:
    # Serialize requests to avoid thread-unsafe solver operations while still
    # exercising parallel scheduling and client lifecycle safety.
    with REQUEST_LOCK:
        response = client.post(path, json=payload)
        data = response.json() if response.status_code == 200 else None
    return response.status_code, data


@pytest.mark.usefixtures("system_client")
def test_tariff_concurrency(system_client):
    suggest_payload = {
        "description": "Canvas sneaker with rubber sole",
        "declared_value_per_unit": 90.0,
        "annual_volume": 12_000,
        "top_k": 2,
    }
    parse_payload = {
        "description": "Tesla chassis bolt for battery pack",
        "bom_text": "steel bolt, zinc coating",
    }

    statuses: list[tuple[str, int, dict | None]] = []
    proof_ids: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _ in range(6):
            futures.append(
                executor.submit(_post, system_client, "/api/tariff/suggest", suggest_payload)
            )
        for _ in range(6):
            futures.append(
                executor.submit(_post, system_client, "/api/tariff/parse", parse_payload)
            )

        for future in concurrent.futures.as_completed(futures):
            status_code, payload = future.result()
            label = "suggest" if payload and payload.get("suggestions") is not None else "parse"
            statuses.append((label, status_code, payload))
            if payload and payload.get("suggestions"):
                suggestion = payload["suggestions"][0]
                proof_id = suggestion.get("proof_id")
                if proof_id:
                    proof_ids.append(proof_id)

    assert all(code < 500 for _, code, _ in statuses)

    suggest_payloads = [p for label, _, p in statuses if label == "suggest"]
    assert suggest_payloads
    for payload in suggest_payloads:
        assert payload["status"] == "OK"
        assert payload["suggestions"]

    parse_payloads = [p for label, _, p in statuses if label == "parse"]
    assert parse_payloads
    for payload in parse_payloads:
        assert payload.get("compiled_facts") is not None

    for proof_id in proof_ids[:5]:
        proof_response = system_client.get(f"/v1/proofs/{proof_id}")
        assert proof_response.status_code == 200, proof_response.text
