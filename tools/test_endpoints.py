"""Smoke-test CALE API endpoints to ensure consistent metric payloads."""

from __future__ import annotations

import json
import signal
import subprocess
import time
from typing import Dict, Tuple

import requests

URL_BASE = "http://127.0.0.1:8000/v1/law"
ENDPOINTS = ["analyze", "suggest_amendment", "train/dry_run"]

TEST_PAYLOAD: Dict[str, Dict[str, object]] = {
    "rule1": {
        "text": "An organization MUST obtain consent before collecting personal data.",
        "jurisdiction": "Canada",
        "statute": "PIPEDA",
        "section": "7(3)",
        "enactment_year": 2000,
    },
    "rule2": {
        "text": "A government agency MAY collect personal data without consent in cases of national security.",
        "jurisdiction": "Canada",
        "statute": "Anti-Terrorism Act",
        "section": "83.28",
        "enactment_year": 2001,
    },
}


def run_server() -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [
            "uvicorn",
            "potatobacon.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(20):
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError("API server exited during startup")
        try:
            requests.get("http://127.0.0.1:8000/docs", timeout=0.5)
            break
        except requests.RequestException:
            continue
    return proc


def stop_server(proc: subprocess.Popen[str]) -> None:
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=3)
    except Exception:
        proc.kill()


def check(endpoint: str) -> Tuple[int, str]:
    response = requests.post(f"{URL_BASE}/{endpoint}", json=TEST_PAYLOAD, timeout=20)
    return response.status_code, response.text


def validate(body: str) -> bool:
    try:
        data = json.loads(body)
    except Exception:
        return False

    required = {
        "conflict_intensity",
        "semantic_overlap",
        "temporal_drift",
        "authority_balance",
        "ccs_scores",
        "suggested_amendment",
    }
    return required.issubset(data)


if __name__ == "__main__":
    process = run_server()
    results = {}
    try:
        for endpoint in ENDPOINTS:
            status, body = check(endpoint)
            results[endpoint] = {
                "status": status,
                "ok": status == 200 and validate(body),
                "body": body[:400],
            }
    finally:
        stop_server(process)

    print(json.dumps(results, indent=2))
