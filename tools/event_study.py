#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rapid CALE event-study validator for leverage/debt conflicts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

try:  # optional dependency
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - fallback if sklearn missing
    roc_auc_score = None  # type: ignore[assignment]

try:  # optional dependency
    from scipy import stats
except Exception:  # pragma: no cover - fallback if scipy missing
    stats = None  # type: ignore[assignment]

from tools import finance_extract as fx


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(y_true, scores))
        except ValueError:
            return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[y_true == 1]
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.mean() - (n_pos - 1) / 2.0) / n_neg)


def welch_pvalue(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan")
    if stats is not None:
        return float(stats.ttest_ind(sample_a, sample_b, equal_var=False).pvalue)
    mean_a, mean_b = float(sample_a.mean()), float(sample_b.mean())
    var_a, var_b = float(sample_a.var(ddof=1)), float(sample_b.var(ddof=1))
    n_a, n_b = len(sample_a), len(sample_b)
    numerator = mean_a - mean_b
    denominator = math.sqrt(var_a / n_a + var_b / n_b)
    if denominator == 0.0:
        return float("nan")
    t_stat = numerator / denominator
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a**2) / ((n_a**2) * (n_a - 1)) + (var_b**2) / ((n_b**2) * (n_b - 1))
    dof = df_num / df_den if df_den != 0 else min(n_a, n_b) - 1
    return float(2.0 * 0.5 * math.erfc(abs(t_stat) / math.sqrt(2)))


def confusion_matrix(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def write_evidence(
    evidence_store: List[Dict[str, object]],
    ticker: str,
    best_row: Dict[str, object],
    evidence_rows: Sequence[Dict[str, object]],
) -> None:
    samples = fx.top_evidence_sentences(evidence_rows, limit=6)
    for ev in samples:
        evidence_store.append(
            {
                "ticker": ticker,
                "filing": ev.get("filing"),
                "sentence": ev.get("sentence"),
                "cce_raw": float(ev.get("cce_raw", 0.0)),
                "cce_delta": float(best_row.get("cce_delta", 0.0)),
                "weight": float(ev.get("weight", 1.0)),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--controls-csv", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--user-agent", required=True)
    parser.add_argument("--out-dir", default="reports/leverage_alpha")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_csv)
    controls = pd.read_csv(args.controls_csv)

    rows: List[Dict[str, object]] = []
    evidence_entries: List[Dict[str, object]] = []
    top_pairs: List[Dict[str, object]] = []

    # Distressed samples
    for _, record in events.iterrows():
        ticker = str(record["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(record["event_date"]))
        best_row, evidence_rows, _ = fx.extract_filing_features(ticker, as_of, args.user_agent, args.api_base)
        if best_row is None:
            continue
        feature_row = {k: v for k, v in best_row.items() if k not in {"cue_meta"}}
        feature_row["label"] = 1
        rows.append(feature_row)
        write_evidence(evidence_entries, ticker, best_row, evidence_rows)
        top_pairs.extend(evidence_rows)

    # Control samples
    for _, record in controls.iterrows():
        ticker = str(record["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(record["event_date"]))
        best_row, evidence_rows, _ = fx.extract_filing_features(ticker, as_of, args.user_agent, args.api_base)
        if best_row is None:
            continue
        feature_row = {k: v for k, v in best_row.items() if k not in {"cue_meta"}}
        feature_row["label"] = 0
        rows.append(feature_row)
        write_evidence(evidence_entries, ticker, best_row, evidence_rows)
        top_pairs.extend(evidence_rows)

    if not rows:
        print("No rows extracted. Check SEC rate limits, API availability, and CSV inputs.")
        return

    df = pd.DataFrame(rows)
    feature_path = out_dir / "event_scores.csv"
    df.to_csv(feature_path, index=False)

    evidence_path = out_dir / "evidence.csv"
    evidence_df = pd.DataFrame(evidence_entries)
    evidence_df.to_csv(evidence_path, index=False)

    top_pairs_path = out_dir / "top_pairs.json"
    with top_pairs_path.open("w", encoding="utf-8") as handle:
        json.dump(top_pairs[:50], handle, indent=2)

    y = df["label"].to_numpy(dtype=float)
    scores = df["CCE"].to_numpy(dtype=float)
    auc = compute_auc(y, scores)

    distressed = df[df.label == 1]["CCE"].to_numpy(dtype=float)
    control = df[df.label == 0]["CCE"].to_numpy(dtype=float)
    p_value = welch_pvalue(distressed, control)
    cm = confusion_matrix(y, scores)

    print("\n=== CALE Event Study (CCE Baseline) ===")
    print(f"N(distressed)={len(distressed)}, mean CCE={float(distressed.mean()) if len(distressed) else float('nan'):.3f}")
    print(f"N(control)   ={len(control)}, mean CCE={float(control.mean()) if len(control) else float('nan'):.3f}")
    print(f"AUC={auc:.3f}")
    print(f"Welch t-test p-value={p_value:.4f}")
    print(f"Confusion matrix (thr=0.5): {cm}")
    print(f"Saved features → {feature_path}")
    print(f"Saved evidence → {evidence_path}")
    print(f"Saved pairs → {top_pairs_path}")
    print("Note: Research tool only — not investment advice.")


if __name__ == "__main__":  # pragma: no cover
    np.random.seed(42)
    main()
