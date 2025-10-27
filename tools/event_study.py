#!/usr/bin/env python3
"""Baseline finance event-study using the CALE finance extractor."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from potatobacon.cale import finance_extract


def load_config() -> dict:
    if finance_extract.CFG:
        return dict(finance_extract.CFG)
    return {}


def strip_html(text: str) -> str:
    import re

    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def read_filing(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Filing path not found: {path}")
    return file_path.read_text(encoding="utf-8", errors="ignore")


@dataclass
class FilingResult:
    ticker: str
    label: int
    cce: float
    features: dict
    evidence: List[dict]


class Logistic:
    def __init__(self, lr: float = 0.05, max_iter: int = 2000, l2: float = 0.0, seed: int = 42):
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.max_iter):
            z = X @ self.w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = (X.T @ (p - y)) / n + self.l2 * self.w
            self.w -= self.lr * grad
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        z = X @ self.w
        return 1.0 / (1.0 + np.exp(-z))


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos = ranks[y_true == 1]
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (pos.mean() - (n_pos - 1) / 2.0) / n_neg


def welch_ttest(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    nx, ny = len(x), len(y)
    denom = math.sqrt(vx / nx + vy / ny)
    if denom == 0:
        return float("nan")
    t_stat = (mx - my) / denom
    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx * vx) / (nx * nx * (nx - 1)) + (vy * vy) / (ny * ny * (ny - 1))
    if df_den == 0:
        return float("nan")
    df = df_num / df_den
    # two-sided p-value approximation using survival function of t-dist
    # Normal approximation for two-sided p-value
    z = abs(t_stat)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return float(max(0.0, min(1.0, p)))


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Tuple[int, int, int, int]]:
    thresholds = np.unique(scores)
    best_thr = 0.5
    best_stat = -1
    best_conf = (0, 0, 0, 0)
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tpr = tp / max(1, (y_true == 1).sum())
        fpr = fp / max(1, (y_true == 0).sum())
        stat = tpr - fpr
        if stat > best_stat:
            best_stat = stat
            best_thr = thr
            best_conf = (tp, fp, tn, fn)
    return best_thr, best_conf


def analyse_filing(ticker: str, filing_path: str, label: int, strict: bool) -> FilingResult:
    raw = read_filing(filing_path)
    text = strip_html(raw)
    result = finance_extract.analyse_finance_sections(text, strict=strict)
    features = {
        "ticker": ticker,
        "C": float(result.get("conflict_intensity", 0.0)),
        "Ab": float(result.get("authority_balance", 0.0)),
        "Dv": float(result.get("fragility", 0.0)),
        "B": float(result.get("bypass", 0.0)),
        "S": float(result.get("semantic_overlap", 0.0)),
        "CCE_prod": float(result.get("CCE_prod", 0.0)),
        "CCE_logistic": float(result.get("CCE_logistic", 0.0)),
    }
    evidence = result.get("evidence", []) or []
    return FilingResult(ticker, label, features["CCE_prod"], features, evidence)


def run_study(events_df: pd.DataFrame, controls_df: pd.DataFrame, strict: bool) -> Tuple[pd.DataFrame, List[dict]]:
    rows = []
    evidence_dump: List[dict] = []
    for _, row in events_df.iterrows():
        res = analyse_filing(str(row["ticker"]).upper(), row["filing_path"], 1, strict)
        rows.append({"label": 1, **res.features})
        if res.evidence:
            ev = dict(res.evidence[0])
            ev["ticker"] = res.ticker
            evidence_dump.append(ev)
    for _, row in controls_df.iterrows():
        res = analyse_filing(str(row["ticker"]).upper(), row["filing_path"], 0, strict)
        rows.append({"label": 0, **res.features})
        if res.evidence:
            ev = dict(res.evidence[0])
            ev["ticker"] = res.ticker
            evidence_dump.append(ev)
    df = pd.DataFrame(rows)
    return df, evidence_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--controls-csv", required=True)
    parser.add_argument("--api-base", default="")  # legacy flag, unused
    parser.add_argument("--user-agent", default="CALE-Research/0.1")
    parser.add_argument("--out-dir", default="reports/leverage_alpha")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    events_df = pd.read_csv(args.events_csv)
    controls_df = pd.read_csv(args.controls_csv)

    strict_mode = False
    df, evidence = run_study(events_df, controls_df, strict_mode)
    if df["CCE_prod"].sum() == 0.0 and not strict_mode:
        strict_mode = True
        df, evidence = run_study(events_df, controls_df, strict_mode)

    csv_path = Path(args.out_dir) / "event_scores.csv"
    df.to_csv(csv_path, index=False)
    with (Path(args.out_dir) / "top_pairs.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence[:50], handle, indent=2)

    y = df["label"].to_numpy()
    scores = df["CCE_prod"].to_numpy()
    auc_base = roc_auc_score(y, scores)

    # Logistic on [C, Ab, Dv, B]
    X = df[["C", "Ab", "Dv", "B"]].to_numpy(dtype=float)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    model = Logistic(lr=0.05, max_iter=1500, l2=0.1)
    model.fit(X, y)
    logits = model.predict(X)
    auc_log = roc_auc_score(y, logits)

    distress = df[df.label == 1]["CCE_prod"].to_numpy()
    control = df[df.label == 0]["CCE_prod"].to_numpy()
    t_p = welch_ttest(distress, control)

    thr, conf = youden_threshold(y, scores)

    print("=== Baseline Finance Event Study ===")
    print(f"Strict mode: {strict_mode}")
    print(f"N(distressed)={len(distress)}, mean={float(distress.mean() if len(distress) else float('nan')):.3f}")
    print(f"N(control)   ={len(control)}, mean={float(control.mean() if len(control) else float('nan')):.3f}")
    print(f"Baseline AUC (CCE_prod): {auc_base:.3f}")
    print(f"Logistic AUC (C,Ab,Dv,B): {auc_log:.3f}")
    print(f"T-test p-value: {t_p:.4f}")
    print(f"Confusion threshold={thr:.3f} -> TP={conf[0]}, FP={conf[1]}, TN={conf[2]}, FN={conf[3]}")
    print(f"Saved scores: {csv_path}")
    print(f"Saved evidence: {Path(args.out_dir) / 'top_pairs.json'}")


if __name__ == "__main__":
    main()
