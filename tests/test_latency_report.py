import json
from pathlib import Path


def test_latency_summary_json_is_valid():
    path = Path("reports/audit/latency_summary.json")
    assert path.exists(), "latency summary artifact is missing"

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert isinstance(payload, dict)
    assert payload != {}
