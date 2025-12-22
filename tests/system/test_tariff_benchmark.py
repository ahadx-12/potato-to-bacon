import json
from pathlib import Path

import pytest

from scripts.run_tariff_benchmark import run_benchmark, _load_benchmark_records


@pytest.mark.usefixtures("system_client")
def test_benchmark_baseline_assigned_rate_and_determinism(tmp_path):
    benchmark_path = tmp_path / "bench.jsonl"
    records = [
        {
            "sku_id": "BM-USB-CABLE-TST",
            "description": "USB-C cable assembly with molded plugs",
            "origin_country": "VN",
            "unit_value": 3.0,
            "annual_volume": 100000,
            "expected_top3_hts": ["HTS_ELECTRONICS_SIGNAL_LOW_VOLT", "HTS_ELECTRONICS_CONNECTOR"],
        },
    ]
    with benchmark_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    results = run_benchmark(benchmark_path=benchmark_path)
    assert results["baseline_assigned_rate"] == 1.0
    assert results["payload_hashes"]

    # determinism check: rerun and compare hashes
    results_two = run_benchmark(benchmark_path=benchmark_path)
    assert results["payload_hashes"] == results_two["payload_hashes"]

    loaded = _load_benchmark_records(benchmark_path)
    assert loaded[0]["sku_id"] == "BM-USB-CABLE-TST"
