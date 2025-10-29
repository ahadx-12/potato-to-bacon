#!/usr/bin/env python3
"""Post-processing helper for the offline CALE event study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))

import tools.event_study_core as core


def _load_metrics(out_dir: Path) -> Dict[str, object]:
    path = out_dir / "metrics.json"
    if not path.exists():
        raise SystemExit(
            "metrics.json missing – run tools/event_study.py before the delta variant"
        )
    return json.loads(path.read_text())


def _load_event_scores(out_dir: Path) -> "pd.DataFrame":
    pd = core.ensure_pandas()
    path = out_dir / "event_scores.csv"
    if not path.exists():
        raise SystemExit(
            "event_scores.csv missing – run tools/event_study.py before the delta variant"
        )
    return pd.read_csv(path)


def _apply_logistic(scores: "pd.DataFrame", meta: Dict[str, object]) -> List[float]:
    names = meta.get("feature_names", [])
    weights = np.array(meta.get("weights", []), dtype=float)
    mean = np.array(meta.get("mean", []), dtype=float)
    std = np.array(meta.get("std", []), dtype=float)
    if not names or len(weights) != len(names):
        raise SystemExit("Invalid logistic metadata in metrics.json")
    if scores.empty:
        return []
    cols = {name: scores.get(name) for name in names if name != "bias"}
    X = []
    for _, row in scores.iterrows():
        vec = [1.0]
        for name in names[1:]:
            vec.append(float(row.get(name, 0.0)))
        X.append(vec)
    X_arr = np.array(X, dtype=float)
    if mean.size and std.size and X_arr.shape[1] == len(weights):
        X_arr[:, 1:] = (X_arr[:, 1:] - mean) / (std + 1e-6)
    logits = X_arr @ weights
    return list(1.0 / (1.0 + np.exp(-logits)))


def run_delta(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    metrics = _load_metrics(out_dir)
    scores = _load_event_scores(out_dir)
    logistic_meta = metrics.get("logistic_model", {})
    logistic_probs = _apply_logistic(scores, logistic_meta)
    delta_score = np.clip(
        scores["CCE"].astype(float) - scores["cce_raw"].astype(float) + 0.05 * scores["severity_hits"],
        0.0,
        1.0,
    )

    scores_delta = scores.copy()
    scores_delta["logistic_probability"] = logistic_probs
    scores_delta["delta_score"] = delta_score
    scores_delta.to_csv(out_dir / "event_scores_delta.csv", index=False)

    coeffs_path = out_dir / "logistic_coeffs.json"
    with coeffs_path.open("w", encoding="utf-8") as handle:
        json.dump(logistic_meta, handle, indent=2)

    top_pairs_src = out_dir / "top_pairs.json"
    if top_pairs_src.exists():
        (out_dir / "top_pairs_delta.json").write_text(top_pairs_src.read_text())

    return {
        "logistic_coeffs": coeffs_path,
        "rows": len(scores_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    core.configure_cli(parser)
    args = parser.parse_args()
    summary = run_delta(args)
    print(
        "ΔCCE artefacts refreshed (rows={rows}, coeffs={logistic_coeffs})".format(
            **{k: str(v) for k, v in summary.items()}
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

