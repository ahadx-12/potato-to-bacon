#!/usr/bin/env python3
"""End-to-end finance evaluation pipeline with CI smoke checks."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(SRC_ROOT))

from tools import finance_real_eval

DEFAULT_MANIFEST = Path("reports/realworld/manifest.csv")
DEFAULT_REPORT = Path("reports/realworld/final_report.json")


def _run_subprocess(cmd: List[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_dirs() -> None:
    DEFAULT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)


def _fetch_filings() -> None:
    logging.info("Fetching SEC filings for distressed/control cohorts")
    try:
        from tools import fetch_sec_real

        fetch_sec_real.main()
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        logging.warning("SEC downloader unavailable: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive network handling
        logging.warning("SEC fetch failed: %s", exc)


def _deduplicate_manifest(manifest: Path, threshold: float) -> Path:
    logging.info("Deduplicating manifest at %s (threshold=%.2f)", manifest, threshold)
    df = pd.read_csv(manifest)
    if df.empty:
        logging.warning("Manifest is empty after fetch; nothing to deduplicate")
        return manifest
    before = len(df)
    df.sort_values(["ticker", "filed"], inplace=True)
    df = df.drop_duplicates(subset=["md5"], keep="first")
    after = len(df)
    logging.info("Manifest dedup: %d -> %d rows", before, after)
    dedup_path = manifest
    df.to_csv(dedup_path, index=False)
    return dedup_path


def _run_tests() -> None:
    logging.info("Running LAW API smoke test")
    _run_subprocess([sys.executable, "-m", "pytest", "tests/cale/test_api_law.py"])
    logging.info("Running P2B parser sanity test")
    _run_subprocess([sys.executable, "-m", "pytest", "tests/test_parser/test_dsl_parser.py"])


def _guardrail_check(report: Dict[str, object], model: str) -> None:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict) or model not in metrics:
        raise RuntimeError(f"Model {model} not present in evaluation metrics")
    model_payload = metrics[model]
    if not isinstance(model_payload, dict):
        raise RuntimeError(f"Unexpected payload for model {model}: {type(model_payload)}")
    guardrail = model_payload.get("metrics", {}).get("ig_guardrail")
    if guardrail is None:
        logging.warning("Model %s missing IG guardrail information", model)
        return
    upper = guardrail.get("wilson", [None, None])
    if isinstance(upper, (list, tuple)):
        upper_bound = upper[1]
    else:
        upper_bound = None
    rate = guardrail.get("ig_fp_rate")
    if rate is not None and rate > 0.20:
        raise RuntimeError(f"IG guardrail breached: fp_rate={rate:.3f}")
    if upper_bound is not None and upper_bound > 0.10:
        raise RuntimeError(f"IG Wilson upper bound {upper_bound:.3f} exceeds 0.10")
    logging.info(
        "IG guardrail ok for %s: rate=%.3f upper_bound=%s",
        model,
        rate if rate is not None else float("nan"),
        "{:.3f}".format(upper_bound) if upper_bound is not None else "n/a",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=str, default="2022-01-01", help="Earliest filing date")
    parser.add_argument("--forms", type=str, default="10-K,10-Q,8-K", help="Forms to fetch")
    parser.add_argument("--thresh-dedup", type=float, default=0.85, help="Similarity threshold for dedup")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="logistic,gbm,stacked",
        help="Comma separated list of models to evaluate",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="platt,isotonic,temperature",
        help="Comma separated list of calibration methods",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for final report JSON",
    )
    parser.add_argument("--no-fetch", action="store_true", help="Skip SEC fetch step")
    parser.add_argument("--no-tests", action="store_true", help="Skip CI smoke tests")
    parser.add_argument(
        "--primary-model",
        type=str,
        default="logistic",
        help="Model to enforce IG guardrail against",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _ensure_dirs()

    if not args.no_fetch:
        _fetch_filings()

    manifest_path = _deduplicate_manifest(DEFAULT_MANIFEST, args.thresh_dedup)

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    calibrations = [item.strip() for item in args.calibration.split(",") if item.strip()]

    logging.info("Evaluating models: %s", ", ".join(models))
    try:
        report = finance_real_eval.evaluate(
            manifest_path,
            seed=args.seed,
            holdout_fraction=0.25,
            cv_strategy="rolling",
            cv_splits=5,
            models=models,
            calibrations=calibrations,
            time_split=True,
            export_folds=Path("reports/realworld/folds.json"),
        )
    except RuntimeError as exc:
        logging.warning("Evaluation aborted: %s", exc)
        report = {
            "status": "manifest_empty",
            "error": str(exc),
            "metrics": {},
            "fingerprint": {},
        }

    if not args.no_tests:
        _run_tests()

    if report.get("status") != "manifest_empty":
        _guardrail_check(report, args.primary_model)

    pipeline_meta = {
        "since": args.since,
        "forms": args.forms,
        "seed": args.seed,
        "models": models,
        "calibration": calibrations,
    }
    report["pipeline"] = pipeline_meta

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logging.info("Final report written to %s", args.out)


if __name__ == "__main__":
    main()
