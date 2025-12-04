"""Alpha/entropy coverage and latency harness tests."""

import json
import random
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pytest
from fastapi.testclient import TestClient

REPORT_DIR = Path("reports/audit")
ALPHA_SUMMARY_PATH = REPORT_DIR / "alpha_entropy_summary.json"
LATENCY_SUMMARY_PATH = REPORT_DIR / "latency_summary.json"


def _law_request_payload() -> Dict[str, Dict[str, object]]:
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


def _compute_r_alpha(metrics: Dict[str, float]) -> float | None:
    value = float(metrics.get("value", 0.0))
    entropy = float(metrics.get("entropy", 0.0))
    risk = float(metrics.get("risk", 0.0))
    denom = entropy * risk
    if denom <= 1e-9:
        return None
    return value / denom


def _median(values: Iterable[float]) -> float:
    sequence: List[float] = list(values)
    if not sequence:
        return 0.0
    return float(statistics.median(sequence))


def test_arbitrage_alpha_entropy_summary(system_client: TestClient):
    random.seed(42)
    runs: Sequence[Sequence[str]] = [
        ["US"],
        ["IE"],
        ["KY"],
        ["US", "IE", "KY"],
    ]

    single_r_alphas: List[float] = []
    multi_r_alphas: List[float] = []
    summary_runs = []

    for jurisdictions in runs:
        response = system_client.post("/api/law/arbitrage/hunt", json={"jurisdictions": jurisdictions})
        assert response.status_code == 200, response.text
        metrics = response.json()["metrics"]
        r_alpha = _compute_r_alpha(metrics)
        summary_runs.append({
            "jurisdictions": jurisdictions,
            "metrics": metrics,
            "r_alpha": r_alpha,
        })

        if len(jurisdictions) == 1 and r_alpha is not None:
            single_r_alphas.append(r_alpha)
        if len(jurisdictions) > 1 and r_alpha is not None:
            multi_r_alphas.append(r_alpha)

    assert single_r_alphas, "Expected at least one single-jurisdiction baseline"
    assert multi_r_alphas, "Expected at least one multi-jurisdiction R_alpha value"

    best_single = max(single_r_alphas)
    best_multi = max(multi_r_alphas)
    assert best_multi > best_single

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ALPHA_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "runs": summary_runs,
                "best_single_r_alpha": best_single,
                "best_multi_r_alpha": best_multi,
            },
            indent=2,
        )
    )


def _median_latency(client: TestClient, path: str, payload: Dict[str, object], runs: int) -> float:
    durations: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        durations.append((time.perf_counter() - start) * 1000.0)
    return _median(durations)


def _job_latency(client: TestClient, payload: Dict[str, object], runs: int) -> float:
    durations: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        kickoff = client.post("/api/law/arbitrage/hunt/job", json=payload)
        assert kickoff.status_code == 200, kickoff.text
        job_id = kickoff.json()["job_id"]

        final_state = None
        for _ in range(30):
            state = client.get(f"/api/law/jobs/{job_id}")
            assert state.status_code == 200, state.text
            body = state.json()
            if body.get("result"):
                final_state = body
                break
            time.sleep(0.05)

        assert final_state and final_state.get("result"), "Job did not complete"
        durations.append((time.perf_counter() - start) * 1000.0)
    return _median(durations)


def test_latency_medians(system_client: TestClient):
    analyze_payload = _law_request_payload()
    hunt_payload = {"jurisdictions": ["US"]}

    analyze_median = _median_latency(system_client, "/v1/law/analyze", analyze_payload, runs=5)
    hunt_median = _median_latency(system_client, "/api/law/arbitrage/hunt", hunt_payload, runs=3)
    async_median = _job_latency(system_client, hunt_payload, runs=1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LATENCY_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "analyze_ms_median": analyze_median,
                "sync_hunt_ms_median": hunt_median,
                "async_hunt_ms_median": async_median,
            },
            indent=2,
        )
    )

    assert analyze_median > 0
    assert hunt_median > 0
    assert async_median > 0
