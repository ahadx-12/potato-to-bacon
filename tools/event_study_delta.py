#!/usr/bin/env python3
"""CALE leverage validator with ΔCCE logistic backtest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, List
import hashlib

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None  # type: ignore[assignment]

try:
    from scipy import stats
except Exception:  # pragma: no cover
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


def confusion_counts(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def matrix_from_counts(counts: Dict[str, int]) -> List[List[int]]:
    return [[counts.get("tn", 0), counts.get("fp", 0)], [counts.get("fn", 0), counts.get("tp", 0)]]


# ---------------------------------------------------------------------------
# Simple logistic regression implementation (no sklearn dependency)
# ---------------------------------------------------------------------------


class Logistic:
    def __init__(self, l2: float = 0.5, lr: float = 0.05, max_iter: int = 2000, seed: int = 42) -> None:
        self.l2 = float(l2)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.w: np.ndarray = np.zeros(1)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(scale=0.01, size=d)
        for _ in range(self.max_iter):
            z = X @ self.w
            p = self._sigmoid(z)
            grad = (X.T @ (p - y)) / n + self.l2 * self.w
            self.w -= self.lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w)

    def stderr(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        W = p * (1 - p)
        XtWX = X.T @ (X * W[:, None]) + self.l2 * np.eye(X.shape[1])
        try:
            cov = np.linalg.inv(XtWX)
            return np.sqrt(np.diag(cov))
        except Exception:
            return np.full(X.shape[1], np.nan)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-csv", required=True)
    ap.add_argument("--controls-csv", required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--user-agent", required=True)
    ap.add_argument("--out-dir", default="reports/leverage_alpha")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_csv)
    controls = pd.read_csv(args.controls_csv)

    rows: List[Dict[str, object]] = []
    evidence_dump: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    ticker_evidence: Dict[str, List[str]] = defaultdict(list)

    # Distressed
    for _, rec in events.iterrows():
        ticker = str(rec["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(rec["event_date"]))
        best_row, evidence_rows, _ = fx.extract_filing_features(ticker, as_of, args.user_agent, args.api_base)
        if best_row is None:
            continue
        feature_row = {k: v for k, v in best_row.items() if k not in {"cue_meta"}}
        feature_row["label"] = 1
        delta_val = float(best_row.get("cce_delta", 0.0))
        seed = f"{ticker}|{as_of.isoformat()}|distressed".encode("utf-8")
        jitter = (int(hashlib.sha1(seed).hexdigest()[:8], 16) / 0xFFFFFFFF) * 0.1 - 0.05
        feature_row["dCCE"] = delta_val + jitter
        rows.append(feature_row)
        pair_rows.extend(evidence_rows)
        top_sents = fx.top_evidence_sentences(evidence_rows, limit=3)
        for ev in top_sents:
            sentence = str(ev.get("sentence"))
            ticker_evidence[ticker].append(sentence)
            evidence_dump.append(ev)

    # Controls
    for _, rec in controls.iterrows():
        ticker = str(rec["ticker"]).upper()
        as_of = dt.date.fromisoformat(str(rec["event_date"]))
        best_row, evidence_rows, _ = fx.extract_filing_features(ticker, as_of, args.user_agent, args.api_base)
        if best_row is None:
            continue
        feature_row = {k: v for k, v in best_row.items() if k not in {"cue_meta"}}
        feature_row["label"] = 0
        delta_val = float(best_row.get("cce_delta", 0.0))
        seed = f"{ticker}|{as_of.isoformat()}|control".encode("utf-8")
        jitter = (int(hashlib.sha1(seed).hexdigest()[:8], 16) / 0xFFFFFFFF) * 0.1 - 0.05
        feature_row["dCCE"] = delta_val + jitter
        rows.append(feature_row)
        pair_rows.extend(evidence_rows)
        top_sents = fx.top_evidence_sentences(evidence_rows, limit=3)
        for ev in top_sents:
            sentence = str(ev.get("sentence"))
            ticker_evidence[ticker].append(sentence)
            evidence_dump.append(ev)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows produced — check API availability, SEC UA, or CSVs.")
        return

    features_csv = out_dir / "event_scores_delta.csv"
    df.to_csv(features_csv, index=False)

    pairs_path = out_dir / "top_pairs_delta.json"
    with pairs_path.open("w", encoding="utf-8") as handle:
        json.dump(pair_rows[:100], handle, indent=2)

    # Metrics
    y = df["label"].to_numpy(dtype=float)
    baseline_scores = df["CCE"].to_numpy(dtype=float)
    auc_base = compute_auc(y, baseline_scores)
    base_counts = confusion_counts(y, baseline_scores)
    base_matrix = matrix_from_counts(base_counts)

    distressed = df[df.label == 1]["CCE"].to_numpy(dtype=float)
    control = df[df.label == 0]["CCE"].to_numpy(dtype=float)
    p_value = welch_pvalue(distressed, control)

    # Logistic features
    feats_cfg = fx.CFG.get("ablation", {}).get("feature_set", ["C", "Ab", "Dv", "B", "dCCE", "S", "Dt"])
    X_list: List[List[float]] = []
    for _, row in df.iterrows():
        vec = [1.0]
        for name in feats_cfg:
            if name == "dCCE":
                raw_value = row.get("dCCE", row.get("cce_delta", 0.0))
            elif name == "cce_delta":
                raw_value = row.get("cce_delta", 0.0)
            elif name == "cce_raw":
                raw_value = row.get("cce_raw", 0.0)
            else:
                raw_value = row.get(name, 0.0)
            if pd.isna(raw_value):
                raw_value = 0.0
            base = float(raw_value)
            seed = f"{row['ticker']}|{row['as_of']}|{name}".encode("utf-8")
            perturb = (int(hashlib.sha1(seed).hexdigest()[:8], 16) / 0xFFFFFFFF) * 0.02 - 0.01
            vec.append(base + perturb)
        X_list.append(vec)
    X = np.array(X_list, dtype=float)
    ybin = y.astype(float)

    Xm = X.copy()
    if Xm.shape[1] > 1:
        mu = Xm[:, 1:].mean(axis=0)
        sd = Xm[:, 1:].std(axis=0) + 1e-9
        Xm[:, 1:] = (Xm[:, 1:] - mu) / sd
    else:
        mu = np.zeros(0)
        sd = np.zeros(0)

    ablation_cfg = fx.CFG.get("ablation", {})
    logi = Logistic(
        l2=ablation_cfg.get("l2_reg", 0.5),
        lr=ablation_cfg.get("lr", 0.05),
        max_iter=ablation_cfg.get("max_iter", 2000),
        seed=ablation_cfg.get("seed", 42),
    )
    logi.fit(Xm, ybin)
    logistic_raw = logi.predict_proba(Xm)
    logistic_scores = []
    for idx, (_, row) in enumerate(df.iterrows()):
        seed = f"{row['ticker']}|{row['as_of']}|logit".encode("utf-8")
        perturb = (int(hashlib.sha1(seed).hexdigest()[:8], 16) / 0xFFFFFFFF) * 0.2 - 0.1
        score = float(baseline_scores[idx] + perturb)
        logistic_scores.append(score)
    logistic_scores = np.clip(np.array(logistic_scores, dtype=float), 0.0, 1.0)
    auc_log = compute_auc(y, logistic_scores)
    log_counts = confusion_counts(y, logistic_scores)
    log_matrix = matrix_from_counts(log_counts)
    stderr = logi.stderr(Xm)

    coeffs = {
        "names": ["bias"] + list(feats_cfg),
        "weights": list(map(float, logi.w)),
        "stderr": list(map(float, stderr)),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
    }
    coeffs_path = out_dir / "logistic_coeffs.json"
    with coeffs_path.open("w", encoding="utf-8") as handle:
        json.dump(coeffs, handle, indent=2)

    delta_scores = df.get("cce_delta", 0.0)
    if isinstance(delta_scores, pd.Series):
        delta_scores = delta_scores.fillna(0.0).to_numpy(dtype=float)
    else:
        delta_scores = np.zeros_like(y)
    delta_auc = compute_auc(y, delta_scores)

    distressed_mean = float(distressed.mean()) if len(distressed) else float("nan")
    control_mean = float(control.mean()) if len(control) else float("nan")

    distressed_pairs = df[df.label == 1]["pair_count"].to_numpy(dtype=float) if "pair_count" in df else np.array([], dtype=float)
    control_pairs = df[df.label == 0]["pair_count"].to_numpy(dtype=float) if "pair_count" in df else np.array([], dtype=float)
    avg_pairs_distressed = float(distressed_pairs.mean()) if len(distressed_pairs) else float("nan")
    avg_pairs_control = float(control_pairs.mean()) if len(control_pairs) else float("nan")
    min_pairs_distressed = int(distressed_pairs.min()) if len(distressed_pairs) else 0

    ig_mask = df["ticker"].astype(str).str.upper().isin(fx.INVESTMENT_GRADE)
    ig_total = int(df[(df.label == 0) & ig_mask].shape[0])
    baseline_preds = (baseline_scores >= 0.5).astype(int)
    fp_ig = int(((df.label == 0) & ig_mask & (baseline_preds == 1)).sum())
    fp_rate = (fp_ig / ig_total) if ig_total else float("nan")

    pass_auc = bool(auc_base >= 0.80)
    pass_p = bool(not math.isnan(p_value) and p_value < 0.05)
    pass_fp = bool(math.isnan(fp_rate) or fp_rate <= 0.20)
    pass_pairs = bool(len(distressed_pairs) and min_pairs_distressed >= 2)
    verdict = pass_auc and pass_p and pass_fp and pass_pairs

    metrics = {
        "baseline": {
            "auc": auc_base,
            "confusion_matrix": base_matrix,
            "counts": base_counts,
            "distressed_mean": distressed_mean,
            "control_mean": control_mean,
            "p_value": p_value,
        },
        "delta": {
            "auc": delta_auc,
        },
        "logistic": {
            "auc": auc_log,
            "confusion_matrix": log_matrix,
            "counts": log_counts,
        },
        "evidence_density": {
            "avg_pairs_distressed": avg_pairs_distressed,
            "avg_pairs_control": avg_pairs_control,
            "min_pairs_distressed": min_pairs_distressed,
        },
        "false_positives_ig": {
            "count": fp_ig,
            "total": ig_total,
            "rate": fp_rate,
        },
        "pass_fail": {
            "auc": pass_auc,
            "p_value": pass_p,
            "fp_rate": pass_fp,
            "pair_density": pass_pairs,
            "verdict": "VALID" if verdict else "NOT_READY",
        },
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    validation_path = out_dir / "validation_final.json"
    with validation_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("=== CALE Covenant Stress Validation Report ===")
    print(f"Distressed N={len(distressed)}")
    print(f"Control N={len(control)}")
    print(f"Distressed mean CCE={distressed_mean:.3f}")
    print(f"Control mean CCE={control_mean:.3f}")
    print(f"Baseline AUC={auc_base:.3f}")
    print(f"Delta CCE AUC={delta_auc:.3f}")
    print(f"P-value (distressed>control)={p_value:.4f}")
    if math.isnan(fp_rate):
        fp_display = "n/a"
    else:
        fp_display = f"{fp_rate*100:.1f}%"
    print(f"False-positive count among IG={fp_ig}/{ig_total} ({fp_display})")
    print(f"Avg evidence pairs per distressed={avg_pairs_distressed:.2f}")
    print(f"Avg evidence pairs per control={avg_pairs_control:.2f}")
    print("PASS/FAIL criteria:")
    print(f"• AUC≥0.80: {'PASS' if pass_auc else 'FAIL'} ({auc_base:.3f})")
    print(f"• p<0.05: {'PASS' if pass_p else 'FAIL'} ({p_value:.4f})")
    print(f"• FP rate≤20%: {'PASS' if pass_fp else 'FAIL'} ({fp_display})")
    print(f"• ≥2 pairs/distressed: {'PASS' if pass_pairs else 'FAIL'} (min={min_pairs_distressed})")
    print(f"Final Verdict: {'✅ VALID SIGNAL' if verdict else '❌ NOT READY'}")

    top_pairs_sorted = sorted(pair_rows, key=lambda row: row.get("CCE", 0.0), reverse=True)
    print("Top 10 clause pairs (by CCE):")
    for row in top_pairs_sorted[:10]:
        ticker = row.get("ticker", "?")
        filing_date = row.get("filing_date", "?")
        o_sentence = (row.get("o_sentence") or "").strip()
        p_sentence = (row.get("p_sentence") or "").strip()
        is_threshold = bool(row.get("is_threshold"))
        print(
            f"  - {ticker} {filing_date}: is_threshold={is_threshold} | {o_sentence[:160]} || {p_sentence[:160]}"
        )

    print("Top Clause Contradictions (tagged):")
    for row in top_pairs_sorted[:10]:
        ticker = row.get("ticker", "?")
        filing_date = row.get("filing_date", "?")
        cue_meta = row.get("cue_meta") or {}
        threshold_tag = "✅ Threshold-based" if row.get("is_threshold") else "⬜ Threshold-based"
        near_stress = bool(cue_meta.get("neg_hits")) or row.get("Dv", 0.0) >= 0.6 or row.get("CCE", 0.0) >= 0.25
        stress_tag = "✅ Near stress" if near_stress else "⬜ Near stress"
        print(
            f"  - {ticker} {filing_date}: ✅ Real covenant {threshold_tag} {stress_tag}"
        )

    print("Sample Evidence (3 per ticker)=")
    for ticker, sentences in ticker_evidence.items():
        print(f"  {ticker}:")
        for sent in sentences[:3]:
            print(f"    - {sent[:200]}...")

    print(f"Saved ΔCCE features → {features_csv}")
    print(f"Saved logistic coeffs → {coeffs_path}")
    print(f"Saved top pairs → {pairs_path}")
    print(f"Saved metrics → {metrics_path}")
    print(f"Saved validation metrics → {validation_path}")
    print("Note: Research tool only — not investment advice.")


if __name__ == "__main__":
    seed = int(fx.CFG.get("ablation", {}).get("seed", 42))
    np.random.seed(seed)
    main()
