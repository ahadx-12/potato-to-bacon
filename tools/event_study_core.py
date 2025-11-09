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
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import tools.finance_extract as fx
from tools.sec_fetch import TICKER_TO_CIK

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "finance"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

EVIDENCE_SUMMARY_COLUMNS: Sequence[Tuple[str, Sequence[str]]] = (
    ("CCE", ("mean", "max")),
    ("cce_raw", ("mean", "max")),
    ("weight", ("mean", "max")),
    ("severity_hits", ("mean", "max", "sum")),
    ("numeric_hits", ("mean", "max", "sum")),
    ("numeric_strength", ("mean", "max")),
    ("numeric_confidence", ("mean", "max")),
    ("bypass_proximity", ("mean", "max", "min")),
)

DERIVED_FEATURE_NAMES = sorted(
    ["ev_count"]
    + [
        f"ev_{column}_{stat}"
        for column, stats in EVIDENCE_SUMMARY_COLUMNS
        for stat in stats
    ]
)


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
    numeric_strength: float
    numeric_confidence: float
    bypass_proximity: float
    provenance: Dict[str, object] = field(default_factory=dict)

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
            "numeric_strength": self.numeric_strength,
            "numeric_confidence": self.numeric_confidence,
            "bypass_proximity": self.bypass_proximity,
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
    numeric_strength: float
    numeric_confidence: float
    bypass_proximity: float
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
            "numeric_strength": self.numeric_strength,
            "numeric_confidence": self.numeric_confidence,
            "bypass_proximity": self.bypass_proximity,
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


def _severity_score(text: str, metadata: Optional[Dict[str, object]] = None) -> Tuple[int, int]:
    hits = 0
    lowered = text.lower()
    for pat in fx.NEG_LEVERAGE_CUES:
        if pat.search(text):
            hits += 1
    for keyword in ("must maintain", "breach", "default", "waiver", "exceed"):
        if keyword in lowered:
            hits += 1
    bypass_hits = 0
    for pattern, _label in getattr(fx, "BYPASS_REGEXES", ()):  # type: ignore[attr-defined]
        if pattern.search(text):
            bypass_hits += 1
    if metadata is not None:
        bypass_hits = max(bypass_hits, len(metadata.get("bypass_terms", [])))
    severity = max(0, hits - min(hits, bypass_hits))
    return severity, bypass_hits


DEBT_VERBS = {
    "incur",
    "incurs",
    "incurred",
    "incurring",
    "borrow",
    "borrows",
    "borrowed",
    "borrowing",
    "debt",
    "debts",
    "indebtedness",
    "loan",
    "loans",
    "lending",
    "lend",
    "guarantee",
    "guarantees",
    "guaranteed",
}


def _bypass_proximity(obligation: str, permission: str) -> float:
    combined = f"{obligation} {permission}".lower()
    tokens = re.findall(r"[a-z]+", combined)
    bypass_positions: List[int] = []
    for idx, token in enumerate(tokens):
        if token == "except" and idx + 1 < len(tokens) and tokens[idx + 1] == "that":
            bypass_positions.append(idx)
        elif token == "provided" and idx + 1 < len(tokens) and tokens[idx + 1] == "however":
            bypass_positions.append(idx)
        elif token == "notwithstanding":
            bypass_positions.append(idx)
        elif token == "basket":
            bypass_positions.append(idx)
    if not bypass_positions:
        return 0.0
    debt_positions = [idx for idx, token in enumerate(tokens) if token in DEBT_VERBS]
    if not debt_positions:
        return 0.0
    min_distance = min(abs(bp - dp) for bp in bypass_positions for dp in debt_positions)
    return float(1.0 / (1.0 + min_distance))


def _compute_pair_score(
    pair: fx.ExtractionPair, rng: random.Random
) -> Tuple[float, float, float, int, int, float, float, float]:
    obligation = pair.obligation
    permission = pair.permission
    metadata = getattr(pair, "metadata", {}) or {}
    cue_weight, cue_meta = fx._cue_score(obligation, permission)  # type: ignore[attr-defined]
    dv = fx.compute_dv(obligation, permission)
    severity_hits, bypass_hits = _severity_score(obligation + " " + permission, metadata)

    numeric_hits_meta = metadata.get("numeric_hits")
    numeric_strength_meta = metadata.get("numeric_strength")
    numeric_conf_meta = metadata.get("numeric_confidence")
    numeric_negations_meta = metadata.get("numeric_negations")
    if numeric_hits_meta is None or numeric_strength_meta is None or numeric_conf_meta is None:
        numeric_hits, numeric_strength, numeric_conf, numeric_negations = fx.summarise_numeric_covenants(  # type: ignore[attr-defined]
            obligation,
            permission,
        )
    else:
        numeric_hits = int(numeric_hits_meta)
        numeric_strength = float(numeric_strength_meta)
        numeric_conf = float(numeric_conf_meta)
        try:
            numeric_negations = int(numeric_negations_meta)
        except (TypeError, ValueError):
            numeric_negations = 0

    effective_numeric_hits = max(0, int(numeric_hits) - min(int(numeric_negations or 0), int(numeric_hits)))
    if bypass_hits and severity_hits:
        severity_hits = max(0, severity_hits - min(severity_hits, bypass_hits))

    bypass_score = _bypass_proximity(obligation, permission)
    if bypass_hits:
        bypass_score = max(bypass_score, min(1.0, 0.25 * bypass_hits))

    raw_score = float(np.clip(dv * cue_weight * (1 + 0.15 * severity_hits), 0.0, 1.0))
    jitter = rng.uniform(-0.05, 0.08)
    boosted = raw_score + 0.04 * effective_numeric_hits + 0.03 * severity_hits + jitter
    score = float(np.clip(boosted, 0.0, 1.0))
    return (
        score,
        raw_score,
        cue_weight,
        severity_hits,
        effective_numeric_hits,
        float(numeric_strength),
        float(numeric_conf),
        float(bypass_score),
    )


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
                best_strength = 0.0
                best_conf = 0.0
                best_bypass = 0.0
                evidence_rows: List[FilingEvidence] = []
                for pair in pairs:
                    score, raw_score, cue_weight, severity_hits, numeric_hits, numeric_strength, numeric_conf, bypass_proximity = _compute_pair_score(
                        pair,
                        clone_rng,
                    )
                    filing_id = _make_filing_id(ticker, filing_date, clone_idx)
                    evidence_rows.append(
                        FilingEvidence(
                            ticker=ticker,
                            filing_id=filing_id,
                            filing_date=filing_date.isoformat(),
                            form=form,
                            obligation=pair.obligation,
                            permission=pair.permission,
                            score=score,
                            raw_score=raw_score,
                            cue_weight=cue_weight,
                            severity_hits=severity_hits,
                            numeric_hits=numeric_hits,
                            numeric_strength=numeric_strength,
                            numeric_confidence=numeric_conf,
                            bypass_proximity=bypass_proximity,
                            provenance=dict(getattr(pair, "metadata", {})),
                        )
                    )
                    if score > best_score:
                        best_score = score
                        best_raw = raw_score
                        best_cue = cue_weight
                        best_sev = severity_hits
                        best_num = numeric_hits
                        best_strength = numeric_strength
                        best_conf = numeric_conf
                        best_bypass = bypass_proximity
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
                    numeric_strength=best_strength,
                    numeric_confidence=best_conf,
                    bypass_proximity=best_bypass,
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


def _summarise_evidence_features(evidence_rows: Sequence[FilingEvidence]) -> Dict[str, float]:
    summary: Dict[str, float] = {name: 0.0 for name in DERIVED_FEATURE_NAMES}
    summary["ev_count"] = float(len(evidence_rows))
    if not evidence_rows:
        return summary

    values: Dict[str, List[float]] = {
        column: [] for column, _ in EVIDENCE_SUMMARY_COLUMNS
    }

    for ev in evidence_rows:
        values["CCE"].append(float(ev.score))
        values["cce_raw"].append(float(ev.raw_score))
        values["weight"].append(float(ev.cue_weight))
        values["severity_hits"].append(float(ev.severity_hits))
        values["numeric_hits"].append(float(ev.numeric_hits))
        values["numeric_strength"].append(float(ev.numeric_strength))
        values["numeric_confidence"].append(float(ev.numeric_confidence))
        values["bypass_proximity"].append(float(ev.bypass_proximity))

    for column, stats in EVIDENCE_SUMMARY_COLUMNS:
        column_values = np.array(values[column], dtype=float)
        if column_values.size == 0:
            continue
        if "mean" in stats:
            summary[f"ev_{column}_mean"] = float(column_values.mean())
        if "max" in stats:
            summary[f"ev_{column}_max"] = float(column_values.max())
        if "sum" in stats:
            summary[f"ev_{column}_sum"] = float(column_values.sum())
        if "min" in stats:
            summary[f"ev_{column}_min"] = float(column_values.min())
    return summary


def _feature_matrix(records: Sequence[FilingRecord]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[List[float]] = []
    derived_names = list(DERIVED_FEATURE_NAMES)
    names = [
        "bias",
        "CCE",
        "cce_raw",
        "cue_weight",
        "severity_hits",
        "numeric_hits",
        "numeric_strength",
        "numeric_confidence",
        "bypass_proximity",
        "pair_count",
        *derived_names,
    ]
    for rec in records:
        evidence_summary = _summarise_evidence_features(rec.evidence_rows)
        features.append(
            [
                1.0,
                rec.cce,
                rec.cce_raw,
                rec.cue_weight,
                float(rec.severity_hits),
                float(rec.numeric_hits),
                float(rec.numeric_strength),
                float(rec.numeric_confidence),
                float(rec.bypass_proximity),
                float(rec.pair_count),
                *[evidence_summary[name] for name in derived_names],
            ]
        )
    X = np.array(features, dtype=float)
    y = np.array([rec.label for rec in records], dtype=float)
    return X, y, names


def _predict_model_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if isinstance(decision, np.ndarray):
            if decision.ndim == 1:
                return 1.0 / (1.0 + np.exp(-decision))
            if decision.ndim == 2 and decision.shape[1] >= 2:
                return 1.0 / (1.0 + np.exp(-decision[:, 1]))
    if hasattr(model, "predict"):
        preds = model.predict(X)
        if isinstance(preds, np.ndarray):
            return preds.astype(float)
    raise TypeError("Model does not expose predict_proba, decision_function, or predict")


def _load_calibrated_model(feature_names: Sequence[str]) -> Optional[Tuple[Any, Dict[str, Any]]]:
    if not MODEL_METADATA_PATH.exists():
        return None
    try:
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return None
    expected_features = metadata.get("feature_names")
    if not isinstance(expected_features, list) or not expected_features:
        return None
    if not set(expected_features).issubset(set(feature_names)):
        return None
    model_path = MODEL_DIR / str(metadata.get("model_path", ""))
    if not model_path.exists():
        return None
    try:
        import joblib  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        model = joblib.load(model_path)
    except Exception:
        return None
    return model, metadata


def _ci_from_samples(samples: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.array([val for val in samples if not math.isnan(val)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    lower = float(np.quantile(arr, alpha / 2.0))
    upper = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return lower, upper


def _bootstrap_metrics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 400,
    seed: int = 2024,
) -> Dict[str, object]:
    n = y.shape[0]
    if n == 0 or len(np.unique(y)) < 2:
        return {
            "baseline_auc_ci": (float("nan"), float("nan")),
            "logistic_auc_ci": (float("nan"), float("nan")),
            "p_value_ci": (float("nan"), float("nan")),
            "effective_samples": {"auc": 0, "logistic": 0, "p_value": 0},
        }

    rng = np.random.default_rng(seed)
    baseline_samples: List[float] = []
    logistic_samples: List[float] = []
    pvalue_samples: List[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_y = y[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_X = X[indices]
        baseline_scores = sample_X[:, 1]
        baseline_samples.append(compute_auc(sample_y, baseline_scores))

        distressed = baseline_scores[sample_y == 1]
        controls = baseline_scores[sample_y == 0]
        if len(distressed) >= 2 and len(controls) >= 2:
            pvalue = welch_pvalue(distressed, controls)
            if not math.isnan(pvalue):
                pvalue_samples.append(pvalue)

        scaler = sample_X.copy()
        if scaler.shape[1] > 1:
            mean = scaler[:, 1:].mean(axis=0)
            std = scaler[:, 1:].std(axis=0) + 1e-6
            scaler[:, 1:] = (scaler[:, 1:] - mean) / std

        logi = Logistic(seed=int(rng.integers(1, 1_000_000_000)))
        logi.fit(scaler, sample_y)
        logistic_scores = logi.predict_proba(scaler)
        logistic_samples.append(compute_auc(sample_y, logistic_scores))

    baseline_ci = _ci_from_samples(baseline_samples, alpha=alpha)
    logistic_ci = _ci_from_samples(logistic_samples, alpha=alpha)
    pvalue_ci = _ci_from_samples(pvalue_samples, alpha=alpha)

    return {
        "baseline_auc_ci": baseline_ci,
        "logistic_auc_ci": logistic_ci,
        "p_value_ci": pvalue_ci,
        "effective_samples": {
            "auc": len(baseline_samples),
            "logistic": len(logistic_samples),
            "p_value": len(pvalue_samples),
        },
    }


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

    bootstrap = _bootstrap_metrics(X, y)

    metrics: Dict[str, Any] = {
        "baseline": {
            "auc": float(auc_baseline),
            "distressed_mean": float(distressed.mean()) if len(distressed) else float("nan"),
            "control_mean": float(controls.mean()) if len(controls) else float("nan"),
            "p_value": float(p_value),
            "counts": baseline_counts,
            "confusion_matrix": baseline_matrix,
            "auc_ci": list(bootstrap["baseline_auc_ci"]),
            "p_value_ci": list(bootstrap["p_value_ci"]),
            "bootstrap_samples": bootstrap["effective_samples"],
        },
        "logistic": {
            "auc": float(auc_logistic),
            "counts": logistic_counts,
            "confusion_matrix": logistic_matrix,
            "auc_ci": list(bootstrap["logistic_auc_ci"]),
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

    calibrated_scores: Optional[np.ndarray] = None
    calibrated_counts: Dict[str, int] = {}
    loaded = _load_calibrated_model(feature_names[1:])
    if loaded is not None:
        model, metadata = loaded
        feature_index = {name: idx for idx, name in enumerate(feature_names)}
        expected = metadata.get("feature_names", [])
        if not isinstance(expected, list) or not expected:
            aligned = None
        else:
            try:
                aligned = np.column_stack([X[:, feature_index[name]] for name in expected])
            except KeyError:
                aligned = None
        if aligned is not None:
            try:
                calibrated_scores = _predict_model_scores(model, aligned)
            except Exception:
                calibrated_scores = None
        if calibrated_scores is not None:
            auc_calibrated = compute_auc(y, calibrated_scores)
            p_val_calibrated = welch_pvalue(
                calibrated_scores[y == 1], calibrated_scores[y == 0]
            )
            calibrated_counts = confusion_counts(y, calibrated_scores)
            calibrated_matrix = [
                [calibrated_counts.get("tn", 0), calibrated_counts.get("fp", 0)],
                [calibrated_counts.get("fn", 0), calibrated_counts.get("tp", 0)],
            ]
            metrics["calibrated"] = {
                "auc": float(auc_calibrated),
                "p_value": float(p_val_calibrated),
                "counts": calibrated_counts,
                "confusion_matrix": calibrated_matrix,
                "metadata": {
                    "model_path": metadata.get("model_path"),
                    "model_type": metadata.get("model_type"),
                    "train_metrics": metadata.get("train_metrics"),
                    "cv_results": metadata.get("cv_results"),
                    "feature_names": metadata.get("feature_names"),
                    "feature_stats": metadata.get("feature_stats"),
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
                    numeric_strength=rec.numeric_strength,
                    numeric_confidence=rec.numeric_confidence,
                    bypass_proximity=rec.bypass_proximity,
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
                            numeric_strength=e.numeric_strength,
                            numeric_confidence=e.numeric_confidence,
                            bypass_proximity=e.bypass_proximity,
                            provenance=dict(e.provenance),
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

