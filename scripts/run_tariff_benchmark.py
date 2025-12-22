from __future__ import annotations

"""Benchmark runner for tariff dossier baseline accuracy and determinism."""

import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT / "src"))

from potatobacon.proofs.canonical import canonical_json
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID
from potatobacon.tariff.sku_dossier import build_sku_dossier_v2
from potatobacon.tariff.sku_store import SKUStore


BENCHMARK_PATH = Path("data/benchmarks/skus_labeled.jsonl")
LATEST_JSON = Path("reports/benchmarks/latest.json")


def _load_benchmark_records(benchmark_path: Path | None = None) -> List[Dict[str, Any]]:
    target = benchmark_path or BENCHMARK_PATH
    if not target.exists():
        raise FileNotFoundError(target)
    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _payload_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _run_once(rows: List[Dict[str, Any]], store_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    if store_path.exists():
        store_path.unlink()
    sku_store = SKUStore(store_path)
    hits_top3 = 0
    baseline_assigned = 0
    runtimes: List[float] = []
    evidence_density: List[int] = []
    payload_hashes: List[str] = []

    for row in rows:
        sku_id = row["sku_id"]
        sku_store.upsert(
            sku_id,
            {
                "description": row["description"],
                "origin_country": row.get("origin_country"),
                "declared_value_per_unit": row.get("unit_value"),
                "annual_volume": row.get("annual_volume"),
            },
        )
        started = time.perf_counter()
        dossier = build_sku_dossier_v2(
            sku_id,
            law_context=DEFAULT_CONTEXT_ID,
            evidence_requested=False,
            optimize=False,
            store=sku_store,
        )
        runtimes.append(time.perf_counter() - started)
        baseline = dossier.baseline_assigned
        if baseline.atom_id:
            baseline_assigned += 1
        expected_top3 = row.get("expected_top3_hts") or []
        if baseline.atom_id in expected_top3:
            hits_top3 += 1
        payload_hashes.append(_payload_hash(dossier.compiled_facts or {}))
        evidence_density.append(len(dossier.fact_evidence or []))

    total = len(rows)
    baseline_assigned_rate = baseline_assigned / total if total else 0.0
    top3_hit_rate = hits_top3 / total if total else 0.0
    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
    evidence_density_avg = sum(evidence_density) / len(evidence_density) if evidence_density else 0.0

    return (
        {
            "baseline_assigned_rate": baseline_assigned_rate,
            "top3_hit_rate": top3_hit_rate,
            "optimized_rate": 0.0,
            "evidence_density": evidence_density_avg,
            "average_runtime": avg_runtime,
        },
        payload_hashes,
    )


def run_benchmark(benchmark_path: Path | None = None) -> Dict[str, Any]:
    rows = _load_benchmark_records(benchmark_path)
    results_one, hashes_one = _run_once(rows, Path("data/benchmark_skus.jsonl"))
    results_two, hashes_two = _run_once(rows, Path("data/benchmark_skus.jsonl"))

    if results_one["baseline_assigned_rate"] < 1.0:
        raise AssertionError("baseline_assigned_rate below 1.0; update coverage or benchmark data")
    if hashes_one != hashes_two:
        raise AssertionError("Benchmark determinism failed: payload hashes differ between runs")

    combined = {
        **results_one,
        "payload_hashes": hashes_one,
    }
    return combined


def _write_report(results: Dict[str, Any]) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    report_path = reports_dir / f"CALE-TARIFF_BENCHMARK_{timestamp.replace(' ', 'T').replace(':', '')}.md"
    lines = [
        "# CALE-TARIFF Benchmark",
        "",
        f"- Generated: {timestamp}",
        f"- Baseline assigned rate: {results['baseline_assigned_rate']:.3f}",
        f"- Top-3 hit rate: {results['top3_hit_rate']:.3f}",
        f"- Optimized rate: {results['optimized_rate']:.3f}",
        f"- Evidence density: {results['evidence_density']:.3f}",
        f"- Average runtime: {results['average_runtime']:.4f}s",
        f"- Payload hashes: {', '.join(results.get('payload_hashes', []))}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_latest(results: Dict[str, Any]) -> None:
    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(canonical_json(results), encoding="utf-8")


def main() -> None:
    results = run_benchmark()
    report_path = _write_report(results)
    _write_latest(results)
    print(canonical_json({"report": str(report_path), **results}))


if __name__ == "__main__":
    main()
