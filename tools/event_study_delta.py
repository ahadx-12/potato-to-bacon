#!/usr/bin/env python3
"""ΔCCE event-study with logistic combiner for finance filings."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from potatobacon.cale import finance_extract


class Logistic:
    def __init__(self, lr: float = 0.05, max_iter: int = 2000, l2: float = 0.5, seed: int = 42):
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

    def stderr(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        p = self.predict(X)
        W = p * (1 - p)
        XtWX = X.T @ (X * W[:, None]) + self.l2 * np.eye(X.shape[1])
        try:
            cov = np.linalg.inv(XtWX)
            return np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan)


def strip_html(text: str) -> str:
    import re

    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def read_filing(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing filing: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def find_prior(path: Path) -> Path | None:
    company_dir = path.parents[1] if len(path.parents) >= 2 else path.parent
    candidates = sorted(company_dir.glob("**/*.htm"))
    try:
        idx = candidates.index(path)
    except ValueError:
        return None
    if idx == 0:
        return None
    return candidates[idx - 1]


def analyse(path: Path, strict: bool) -> Dict[str, float | List[dict]]:
    raw = read_filing(path)
    text = strip_html(raw)
    return finance_extract.analyse_finance_sections(text, strict=strict)


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


def build_feature_row(ticker: str, label: int, current: Dict[str, float | List[dict]], prior: Dict[str, float | List[dict]] | None) -> Dict[str, float]:
    cce_curr = float(current.get("CCE_prod", 0.0))
    cce_prev = float(prior.get("CCE_prod", 0.0)) if prior else 0.0
    delta = cce_curr - cce_prev
    row = {
        "ticker": ticker,
        "label": label,
        "C": float(current.get("conflict_intensity", 0.0)),
        "Ab": float(current.get("authority_balance", 0.0)),
        "Dv": float(current.get("fragility", 0.0)),
        "B": float(current.get("bypass", 0.0)),
        "S": float(current.get("semantic_overlap", 0.0)),
        "CCE_prod": cce_curr,
        "delta_cce": delta,
        "Dt": 0.0,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--controls-csv", required=True)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--user-agent", default="CALE-Research/0.1")
    parser.add_argument("--out-dir", default="reports/leverage_alpha")
    args = parser.parse_args()

    cfg = finance_extract.CFG or {}
    os.makedirs(args.out_dir, exist_ok=True)

    events = pd.read_csv(args.events_csv)
    controls = pd.read_csv(args.controls_csv)

    def run(strict: bool):
        rows: List[Dict[str, float]] = []
        evidence: List[dict] = []
        for _, row in events.iterrows():
            path = Path(row["filing_path"])
            current = analyse(path, strict)
            prior_path = find_prior(path)
            prior = analyse(prior_path, strict) if prior_path else None
            rows.append(build_feature_row(str(row["ticker"]).upper(), 1, current, prior))
            ev = current.get("evidence", []) or []
            if ev:
                ev0 = dict(ev[0])
                ev0["ticker"] = str(row["ticker"]).upper()
                evidence.append(ev0)
        for _, row in controls.iterrows():
            path = Path(row["filing_path"])
            current = analyse(path, strict)
            prior_path = find_prior(path)
            prior = analyse(prior_path, strict) if prior_path else None
            rows.append(build_feature_row(str(row["ticker"]).upper(), 0, current, prior))
            ev = current.get("evidence", []) or []
            if ev:
                ev0 = dict(ev[0])
                ev0["ticker"] = str(row["ticker"]).upper()
                evidence.append(ev0)
        df = pd.DataFrame(rows)
        return df, evidence

    strict = False
    df, evidence = run(strict)
    if (df["CCE_prod"].std() < 1e-6 or df["CCE_prod"].sum() == 0.0) and not strict:
        strict = True
        df, evidence = run(strict)

    out_csv = Path(args.out_dir) / "event_scores_delta.csv"
    df.to_csv(out_csv, index=False)

    with (Path(args.out_dir) / "top_pairs_delta.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence[:50], handle, indent=2)

    y = df["label"].to_numpy(dtype=float)
    scores = df["CCE_prod"].to_numpy(dtype=float)
    auc_base = roc_auc_score(y, scores)

    feats_cfg = cfg.get("ablation", {}).get("feature_set", ["C", "Ab", "Dv", "B", "delta_cce", "S", "Dt"])
    feat_cols = []
    for f in feats_cfg:
        if f == "Ab":
            feat_cols.append("Ab")
        elif f in df.columns:
            feat_cols.append(f)
        else:
            df[f] = 0.0
            feat_cols.append(f)
    X = df[feat_cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-9
    X_std = (X - mu) / sigma
    X_std = np.hstack([np.ones((X_std.shape[0], 1)), X_std])

    log_cfg = cfg.get("ablation", {})
    model = Logistic(
        lr=float(log_cfg.get("lr", 0.05)),
        max_iter=int(log_cfg.get("max_iter", 2000)),
        l2=float(log_cfg.get("l2_reg", 0.5)),
        seed=int(log_cfg.get("seed", 42)),
    )
    model.fit(X_std, y)
    probs = model.predict(X_std)
    auc_log = roc_auc_score(y, probs)
    se = model.stderr(X_std)

    coeff_path = Path(args.out_dir) / "logistic_coeffs.json"
    with coeff_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "names": ["bias"] + feat_cols,
                "weights": [float(v) for v in model.w],
                "stderr": [float(v) for v in se],
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
            },
            handle,
            indent=2,
        )

    thr, conf = youden_threshold(y, scores)
    distress = df[df.label == 1]["CCE_prod"].to_numpy()
    control = df[df.label == 0]["CCE_prod"].to_numpy()
    t_p = welch_ttest(distress, control)

    verdict = "FAIL"
    if auc_base >= 0.70 or auc_log >= 0.70:
        verdict = "PASS"
    iteration = "regex_fallback" if strict else "first_pass"

    print("=== CALE Finance Validation Report (Target AUC ≥ 0.70) ===")
    print(f"Distressed N: {len(distress)}, mean CCE: {float(distress.mean() if len(distress) else float('nan')):.3f}")
    print(f"Control    N: {len(control)}, mean CCE: {float(control.mean() if len(control) else float('nan')):.3f}")
    print(f"Baseline AUC (CCE_prod): {auc_base:.3f}")
    print(f"ΔCCE Logistic AUC:        {auc_log:.3f}")
    print(f"t-test p-value:           {t_p:.4f}")
    print(f"Confusion @ optimal thr:  TP={conf[0]}, FP={conf[1]}, TN={conf[2]}, FN={conf[3]}")
    print("\nTop evidence (distressed):")
    distressed_tickers = set(df[df.label == 1]["ticker"])
    dist_evidence = [ev for ev in evidence if ev.get("ticker") in distressed_tickers]
    if not dist_evidence:
        print("- No obligation/permission pairs detected; consider tightening extraction rules.")
    for ev in dist_evidence[:3]:
        c_val = float(ev.get('conflict', 0.0))
        ab_val = float(ev.get('authority', 0.0))
        dv_val = float(ev.get('fragility', 0.0))
        cce_est = c_val * ab_val if ab_val else c_val
        print(
            f"- {ev.get('ticker')} {ev.get('section')}: [OBLIGATION ⟂ PERMISSION], "
            f"bypass={ev.get('bypass')}, Ab={ab_val:.2f}, Dv={dv_val:.2f}, C={c_val:.2f}, "
            f"CCE≈{cce_est:.2f}, sentence=\"{ev.get('obligation', '')[:80]}...\""
        )
    print(f"\nIteration: {iteration}")
    print(f"Verdict: {verdict}")
    print("Artifacts:")
    print(f"- {Path(args.out_dir) / 'event_scores_delta.csv'}")
    print(f"- {Path(args.out_dir) / 'event_scores.csv'}")
    print(f"- {Path(args.out_dir) / 'top_pairs_delta.json'}")
    print(f"- {coeff_path}")
    print("===========================================================")
    if verdict == "PASS":
        print("\nNext steps:")
        print("- Expand sample to ≥100 events / 100 controls (2019–2025)")
        print("- Add borrow-fee filter + universe liquidity filter")
        print("- Build monthly short-decile backtest (hedged) with costs")
        print("- Paper trade 8–12 weeks before deploying capital")


if __name__ == "__main__":
    main()
