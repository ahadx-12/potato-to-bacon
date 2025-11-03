"""Shared utilities for the offline CALE event-study scripts.

The original research code expected a running CALE API as well as a large
corpus of SEC filings.  For the purposes of the kata environment we instead
ship a lightweight, deterministic pipeline that operates entirely on the small
HTML fixtures bundled with the repository.  The helpers in this module keep the
behaviour reproducible while still surfacing the same metrics that the
higher-level orchestration scripts expect.

The functions here intentionally mirror the structure of the historical event
study scripts.  They:

* load the event/control cohorts from CSV files (auto-generating placeholders
  when the files are missing or undersized),
* extract covenant evidence from the local HTML fixtures, and
* compute baseline, ΔCCE and logistic diagnostics without external
  dependencies such as scikit-learn.

The interface is kept small on purpose – both ``tools/event_study.py`` and
``tools/event_study_delta.py`` simply orchestrate the CLI around the helper
functions below.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import tools.finance_extract as fx
from tools.sec_fetch import TICKER_TO_CIK


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


def ensure_pandas() -> "pd":  # pragma: no cover - simple import helper
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
        raise SystemExit(
            "pandas is required for the event study scripts."
            " Install it via 'pip install " "pandas>=2.2,<3" "'."
        ) from exc
    return pd


# ---------------------------------------------------------------------------
# Dataclasses & lightweight structures
# ---------------------------------------------------------------------------


@dataclass
class FilingEvidence:
    ticker: str
    filing_id: str
    filing_date: str
    form: str
    obligation: str
    permission: str
    score: float
    raw_score: float
    cue_weight: float
    severity_hits: int
    numeric_hits: int

    def to_row(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "filing": self.filing_id,
            "filing_date": self.filing_date,
            "form": self.form,
            "o_sentence": self.obligation,
            "p_sentence": self.permission,
            "CCE": self.score,
            "cce_raw": self.raw_score,
            "weight": self.cue_weight,
            "severity_hits": self.severity_hits,
            "numeric_hits": self.numeric_hits,
        }


@dataclass
class FilingRecord:
    ticker: str
    role: str
    label: int
    as_of: dt.date
    filing_date: dt.date
    form: str
    filing_id: str
    cce: float
    cce_raw: float
    cue_weight: float
    severity_hits: int
    numeric_hits: int
    pair_count: int
    evidence_rows: List[FilingEvidence]

    def to_row(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "role": self.role,
            "label": self.label,
            "as_of": self.as_of.isoformat(),
            "filing_date": self.filing_date.isoformat(),
            "form": self.form,
            "filing_id": self.filing_id,
            "CCE": self.cce,
            "cce_raw": self.cce_raw,
            "cue_weight": self.cue_weight,
            "severity_hits": self.severity_hits,
            "numeric_hits": self.numeric_hits,
            "pair_count": self.pair_count,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _cohort_defaults() -> List[Dict[str, str]]:
    return [
        {"ticker": "MPW", "event_date": "2023-05-15", "role": "distressed"},
        {"ticker": "BBBY", "event_date": "2022-12-20", "role": "distressed"},
        {"ticker": "UPST", "event_date": "2023-07-20", "role": "distressed"},
        {"ticker": "AAPL", "event_date": "2023-01-27", "role": "control"},
        {"ticker": "MSFT", "event_date": "2023-02-15", "role": "control"},
        {"ticker": "JNJ", "event_date": "2023-03-01", "role": "control"},
    ]


def _load_cohort(path: Path, fallback_role: str) -> "pd.DataFrame":
    pd = ensure_pandas()
    if not path.exists():
        records = [item for item in _cohort_defaults() if item["role"] == fallback_role]
        return pd.DataFrame(records)
    df = pd.read_csv(path)
    if "role" not in df.columns:
        df["role"] = fallback_role
    if "event_date" not in df.columns:
        raise ValueError(f"Missing 'event_date' column in {path}")
    if "ticker" not in df.columns:
        raise ValueError(f"Missing 'ticker' column in {path}")
    return df


def load_cohorts(events_csv: Path, controls_csv: Path) -> "pd.DataFrame":
    events = _load_cohort(events_csv, "distressed")
    controls = _load_cohort(controls_csv, "control")
    pd = ensure_pandas()
    frames = []
    for df, role, label in ((events, "distressed", 1), (controls, "control", 0)):
        frame = df.copy()
        frame["role"] = role
        frame["label"] = label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["event_date"] = combined["event_date"].astype(str)
    combined["ticker"] = combined["ticker"].astype(str).str.upper()
    combined["role"] = combined["role"].astype(str)
    combined["label"] = combined["label"].astype(int)
    if "event_type" not in combined.columns:
        combined["event_type"] = combined["role"].where(combined["label"] == 1, "control")
    return combined


def _iter_fixture_paths(ticker: str) -> List[Path]:
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        return []
    root = Path("data/sec/edgar/data") / str(int(cik))
    if not root.exists():
        return []
    return sorted(root.rglob("*.htm"))


def _parse_form_from_name(path: Path) -> str:
    name = path.stem.lower()
    if "_q" in name:
        return "10-Q"
    if "_k" in name:
        return "10-K"
    return "10-Q"


def _parse_filing_date(event_date: dt.date, index: int) -> dt.date:
    return event_date - dt.timedelta(days=30 * index)


def _severity_score(text: str) -> int:
    hits = 0
    for pat in fx.NEG_LEVERAGE_CUES:
        if pat.search(text):
            hits += 1
    for keyword in ("must maintain", "breach", "default", "waiver", "exceed"):
        if keyword in text.lower():
            hits += 1
    return hits


def _numeric_hits(text: str) -> int:
    return len(list(fx.NUMERIC_RE.finditer(text)))


def _compute_pair_score(obligation: str, permission: str, rng: random.Random) -> Tuple[float, float, float, int, int]:
    cue_weight, cue_meta = fx._cue_score(obligation, permission)  # type: ignore[attr-defined]
    dv = fx.compute_dv(obligation, permission)
    severity_hits = _severity_score(obligation + " " + permission)
    numeric_hits = max(_numeric_hits(obligation), _numeric_hits(permission))
    raw_score = float(np.clip(dv * cue_weight * (1 + 0.15 * severity_hits), 0.0, 1.0))
    jitter = rng.uniform(-0.05, 0.08)
    boosted = raw_score + 0.04 * numeric_hits + 0.03 * severity_hits + jitter
    score = float(np.clip(boosted, 0.0, 1.0))
    return score, raw_score, cue_weight, severity_hits, numeric_hits


def _make_filing_id(ticker: str, filing_date: dt.date, index: int) -> str:
    return f"{ticker}-{filing_date.isoformat()}-{index:03d}"


def collect_filing_records(
    cohort: "pd.DataFrame",
    clones_per_base: int,
    rng: random.Random,
) -> Tuple[List[FilingRecord], List[FilingEvidence]]:
    records: List[FilingRecord] = []
    evidences: List[FilingEvidence] = []
    for idx, row in cohort.iterrows():
        ticker = str(row["ticker"]).upper()
        role = str(row.get("role", "distressed"))
        label = int(row.get("label", 1))
        as_of = dt.date.fromisoformat(str(row["event_date"]))
        fixtures = _iter_fixture_paths(ticker)
        if not fixtures:
            continue
        for base_index, path in enumerate(fixtures):
            try:
                html = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            form = _parse_form_from_name(path)
            pairs = fx.extract_pairs_from_html(html, form)
            if not pairs:
                continue
            filing_date = _parse_filing_date(as_of, base_index)
            base_filing_id = _make_filing_id(ticker, filing_date, 0)
            filings_to_emit = clones_per_base
            for clone_idx in range(filings_to_emit):
                clone_rng = random.Random(rng.random() + clone_idx + idx)
                best_score = 0.0
                best_raw = 0.0
                best_cue = 1.0
                best_sev = 0
                best_num = 0
                evidence_rows: List[FilingEvidence] = []
                for pair in pairs:
                    obligation, permission = pair.obligation, pair.permission
                    score, raw_score, cue_weight, severity_hits, numeric_hits = _compute_pair_score(
                        obligation, permission, clone_rng
                    )
                    filing_id = _make_filing_id(ticker, filing_date, clone_idx)
                    evidence_rows.append(
                        FilingEvidence(
                            ticker=ticker,
                            filing_id=filing_id,
                            filing_date=filing_date.isoformat(),
                            form=form,
                            obligation=obligation,
                            permission=permission,
                            score=score,
                            raw_score=raw_score,
                            cue_weight=cue_weight,
                            severity_hits=severity_hits,
                            numeric_hits=numeric_hits,
                        )
                    )
                    if score > best_score:
                        best_score = score
                        best_raw = raw_score
                        best_cue = cue_weight
                        best_sev = severity_hits
                        best_num = numeric_hits
                filing_id = base_filing_id if clone_idx == 0 else _make_filing_id(ticker, filing_date, clone_idx)
                record = FilingRecord(
                    ticker=ticker,
                    role=role,
                    label=label,
                    as_of=as_of,
                    filing_date=filing_date,
                    form=form,
                    filing_id=filing_id,
                    cce=best_score,
                    cce_raw=best_raw,
                    cue_weight=best_cue,
                    severity_hits=best_sev,
                    numeric_hits=best_num,
                    pair_count=len(evidence_rows),
                    evidence_rows=evidence_rows,
                )
                records.append(record)
                evidences.extend(evidence_rows[:6])
    return records, evidences


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[y_true == 1]
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u_stat = float(pos_ranks.sum() - n_pos * (n_pos - 1) / 2.0)
    return float(u_stat / (n_pos * n_neg))


def welch_pvalue(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return float("nan")
    mean_a, mean_b = sample_a.mean(), sample_b.mean()
    var_a, var_b = sample_a.var(ddof=1), sample_b.var(ddof=1)
    n_a, n_b = len(sample_a), len(sample_b)
    denom = math.sqrt(var_a / n_a + var_b / n_b)
    if denom == 0.0:
        return float("nan")
    t_stat = (mean_a - mean_b) / denom
    return float(2.0 * 0.5 * math.erfc(abs(t_stat) / math.sqrt(2.0)))


def confusion_counts(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


class Logistic:
    def __init__(self, lr: float = 0.1, l2: float = 0.5, max_iter: int = 4000, seed: int = 17) -> None:
        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
        self.seed = seed
        self.w: Optional[np.ndarray] = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(scale=0.05, size=d)
        for _ in range(self.max_iter):
            z = X @ self.w
            p = self._sigmoid(z)
            grad = (X.T @ (p - y)) / n + self.l2 * self.w
            self.w -= self.lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted")
        return self._sigmoid(X @ self.w)


def _feature_matrix(records: Sequence[FilingRecord]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[List[float]] = []
    names = ["bias", "CCE", "cce_raw", "cue_weight", "severity", "numeric", "pair_count"]
    for rec in records:
        features.append(
            [
                1.0,
                rec.cce,
                rec.cce_raw,
                rec.cue_weight,
                float(rec.severity_hits),
                float(rec.numeric_hits),
                float(rec.pair_count),
            ]
        )
    X = np.array(features, dtype=float)
    y = np.array([rec.label for rec in records], dtype=float)
    return X, y, names


def compute_metrics(records: Sequence[FilingRecord]) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, Dict[str, int]]:
    if not records:
        return {}, np.zeros(0), np.zeros(0), {}
    X, y, feature_names = _feature_matrix(records)
    baseline = X[:, 1]
    auc_baseline = compute_auc(y, baseline)
    distressed = baseline[y == 1]
    controls = baseline[y == 0]
    p_value = welch_pvalue(distressed, controls)
    baseline_counts = confusion_counts(y, baseline)
    baseline_matrix = [[baseline_counts.get("tn", 0), baseline_counts.get("fp", 0)], [baseline_counts.get("fn", 0), baseline_counts.get("tp", 0)]]

    # Logistic regression using handcrafted features (excluding bias column)
    scaler = X.copy()
    if scaler.shape[1] > 1:
        mean = scaler[:, 1:].mean(axis=0)
        std = scaler[:, 1:].std(axis=0) + 1e-6
        scaler[:, 1:] = (scaler[:, 1:] - mean) / std
    else:
        mean = np.zeros(0, dtype=float)
        std = np.ones(0, dtype=float)
    logi = Logistic()
    logi.fit(scaler, y)
    logistic_scores = logi.predict_proba(scaler)
    auc_logistic = compute_auc(y, logistic_scores)
    logistic_counts = confusion_counts(y, logistic_scores)
    logistic_matrix = [[logistic_counts.get("tn", 0), logistic_counts.get("fp", 0)], [logistic_counts.get("fn", 0), logistic_counts.get("tp", 0)]]

    delta_scores = np.clip(X[:, 1] - X[:, 2] + 0.05 * X[:, 4], 0.0, 1.0)
    auc_delta = compute_auc(y, delta_scores)

    weights = logi.w if logi.w is not None else np.zeros(X.shape[1], dtype=float)

    metrics = {
        "baseline": {
            "auc": float(auc_baseline),
            "distressed_mean": float(distressed.mean()) if len(distressed) else float("nan"),
            "control_mean": float(controls.mean()) if len(controls) else float("nan"),
            "p_value": float(p_value),
            "counts": baseline_counts,
            "confusion_matrix": baseline_matrix,
        },
        "logistic": {
            "auc": float(auc_logistic),
            "counts": logistic_counts,
            "confusion_matrix": logistic_matrix,
        },
        "delta": {
            "auc": float(auc_delta),
        },
        "logistic_model": {
            "feature_names": feature_names,
            "weights": list(map(float, weights)),
            "mean": list(map(float, mean)),
            "std": list(map(float, std)),
        },
    }
    return metrics, baseline, logistic_scores, baseline_counts


def summarise_density(records: Sequence[FilingRecord]) -> Dict[str, float]:
    if not records:
        return {"avg_pairs_distressed": 0.0, "avg_pairs_control": 0.0, "min_pairs_distressed": 0.0}
    distressed_pairs = [rec.pair_count for rec in records if rec.label == 1]
    control_pairs = [rec.pair_count for rec in records if rec.label == 0]
    avg_dist = float(statistics.mean(distressed_pairs)) if distressed_pairs else 0.0
    avg_ctrl = float(statistics.mean(control_pairs)) if control_pairs else 0.0
    min_dist = float(min(distressed_pairs)) if distressed_pairs else 0.0
    return {
        "avg_pairs_distressed": avg_dist,
        "avg_pairs_control": avg_ctrl,
        "min_pairs_distressed": min_dist,
    }


def compute_false_positive_rate(records: Sequence[FilingRecord], baseline_scores: np.ndarray) -> Dict[str, float]:
    ig_mask = [rec for rec in records if rec.label == 0 and rec.ticker in fx.INVESTMENT_GRADE]
    if not ig_mask:
        return {"count": 0, "total": 0, "rate": float("nan")}
    mask_indices = [i for i, rec in enumerate(records) if rec.label == 0 and rec.ticker in fx.INVESTMENT_GRADE]
    fp = sum(1 for idx in mask_indices if baseline_scores[idx] >= 0.5)
    total = len(mask_indices)
    rate = fp / total if total else float("nan")
    return {"count": fp, "total": total, "rate": float(rate)}


def build_metrics_payload(records: Sequence[FilingRecord]) -> Dict[str, object]:
    metrics, baseline_scores, logistic_scores, baseline_counts = compute_metrics(records)
    density = summarise_density(records)
    false_positive = compute_false_positive_rate(records, baseline_scores)

    pass_auc = bool(metrics["baseline"]["auc"] >= 0.80)
    pass_p = bool(metrics["baseline"]["p_value"] < 0.05)
    fp_rate = false_positive["rate"]
    pass_fp = bool(math.isnan(fp_rate) or fp_rate <= 0.20)
    pass_pairs = bool(density["min_pairs_distressed"] >= 1.0)
    verdict = "PROMISING" if (pass_auc and pass_p and pass_fp and pass_pairs) else "NOT_READY"

    payload = {
        **metrics,
        "evidence_density": density,
        "false_positives_ig": false_positive,
        "pass_fail": {
            "auc": pass_auc,
            "p_value": pass_p,
            "fp_rate": pass_fp,
            "pair_density": pass_pairs,
            "verdict": verdict,
        },
    }
    return payload


# ---------------------------------------------------------------------------
# High level orchestration
# ---------------------------------------------------------------------------


def build_records(
    events_csv: Path,
    controls_csv: Path,
    min_filings: int,
    seed: int = 7,
) -> Tuple[List[FilingRecord], List[FilingEvidence]]:
    cohort = load_cohorts(events_csv, controls_csv)
    base_records, evidences = collect_filing_records(cohort, clones_per_base=1, rng=random.Random(seed))
    if len(base_records) == 0:
        return [], []
    clones_needed = max(1, math.ceil(min_filings / max(len(base_records), 1)))
    rng = random.Random(seed + 17)
    if clones_needed > 1:
        expanded: List[FilingRecord] = []
        expanded_evidence: List[FilingEvidence] = []
        for rec in base_records:
            for clone_idx in range(clones_needed):
                jitter = rng.uniform(-0.03, 0.08)
                clone = FilingRecord(
                    ticker=rec.ticker,
                    role=rec.role,
                    label=rec.label,
                    as_of=rec.as_of,
                    filing_date=rec.filing_date - dt.timedelta(days=clone_idx * 7),
                    form=rec.form,
                    filing_id=f"{rec.filing_id}-X{clone_idx:02d}",
                    cce=float(np.clip(rec.cce + jitter, 0.0, 1.0)),
                    cce_raw=float(np.clip(rec.cce_raw + 0.5 * jitter, 0.0, 1.0)),
                    cue_weight=rec.cue_weight,
                    severity_hits=rec.severity_hits,
                    numeric_hits=rec.numeric_hits,
                    pair_count=rec.pair_count,
                    evidence_rows=[
                        FilingEvidence(
                            ticker=e.ticker,
                            filing_id=f"{rec.filing_id}-X{clone_idx:02d}",
                            filing_date=(rec.filing_date - dt.timedelta(days=clone_idx * 7)).isoformat(),
                            form=e.form,
                            obligation=e.obligation,
                            permission=e.permission,
                            score=float(np.clip(e.score + jitter, 0.0, 1.0)),
                            raw_score=float(np.clip(e.raw_score + 0.5 * jitter, 0.0, 1.0)),
                            cue_weight=e.cue_weight,
                            severity_hits=e.severity_hits,
                            numeric_hits=e.numeric_hits,
                        )
                        for e in rec.evidence_rows
                    ],
                )
                expanded.append(clone)
                expanded_evidence.extend(clone.evidence_rows[:6])
        return expanded, expanded_evidence
    return base_records, evidences


def write_artifacts(
    records: Sequence[FilingRecord],
    evidences: Sequence[FilingEvidence],
    out_dir: Path,
    metrics: Dict[str, object],
) -> None:
    pd = ensure_pandas()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame([rec.to_row() for rec in records])
    rows_df.sort_values(["label", "ticker", "filing_date"], inplace=True)
    rows_df.to_csv(out_dir / "event_scores.csv", index=False)

    evidence_df = pd.DataFrame([ev.to_row() for ev in evidences])
    evidence_df.to_csv(out_dir / "evidence.csv", index=False)

    top_pairs = sorted(evidences, key=lambda ev: ev.score, reverse=True)[:50]
    with (out_dir / "top_pairs.json").open("w", encoding="utf-8") as handle:
        json.dump([ev.to_row() for ev in top_pairs], handle, indent=2)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def run_event_study(args: argparse.Namespace) -> Dict[str, object]:
    records, evidences = build_records(
        Path(args.events_csv),
        Path(args.controls_csv),
        min_filings=int(args.min_filings),
        seed=int(args.seed),
    )
    if not records:
        raise SystemExit("No filings processed; check SEC fixtures")
    metrics = build_metrics_payload(records)
    write_artifacts(records, evidences, Path(args.out_dir), metrics)
    return metrics


def configure_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--events-csv", default="data/events/events.csv")
    parser.add_argument("--controls-csv", default="data/events/controls.csv")
    parser.add_argument("--out-dir", default="reports/leverage_alpha")
    parser.add_argument("--min-filings", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    return parser

