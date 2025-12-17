import json
import time
from pathlib import Path

import pytest


@pytest.mark.usefixtures("system_client")
def test_tariff_batch_scale_sanity(system_client):
    dataset = json.loads(Path("tests/data/realworld_skus.json").read_text(encoding="utf-8"))
    skus = [
        {
            "sku_id": item["sku_id"],
            "description": item["description"],
            "bom_text": item.get("bom_text"),
            "declared_value_per_unit": item.get("declared_value_per_unit", 100.0),
            "annual_volume": item.get("annual_volume"),
        }
        for item in dataset[:50]
    ]

    payload = {
        "skus": skus,
        "top_k_per_sku": 2,
        "max_results": 30,
        "seed": 909,
        "include_all_suggestions": False,
    }

    start = time.perf_counter()
    response = system_client.post("/api/tariff/batch-scan", json=payload)
    duration_first = time.perf_counter() - start
    assert response.status_code == 200, response.text

    data_first = response.json()
    assert data_first["status"] == "OK"
    assert data_first["total_skus"] == len(skus)
    assert data_first["processed_skus"] == len(skus)

    for result in data_first["results"]:
        assert result["status"] == "OK"
        assert result["best"]
        assert result["best"]["proof_id"]

    start_second = time.perf_counter()
    response_second = system_client.post("/api/tariff/batch-scan", json=payload)
    duration_second = time.perf_counter() - start_second
    assert response_second.status_code == 200, response_second.text
    data_second = response_second.json()

    first_ids = [item["sku_id"] for item in data_first["results"]]
    second_ids = [item["sku_id"] for item in data_second["results"]]
    assert first_ids == second_ids

    print(f"Batch scan durations: first={duration_first:.3f}s second={duration_second:.3f}s")

    for result in data_first["results"][:5]:
        proof_id = result["best"]["proof_id"]
        proof_response = system_client.get(f"/v1/proofs/{proof_id}")
        assert proof_response.status_code == 200, proof_response.text
