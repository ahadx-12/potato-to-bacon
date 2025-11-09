#!/usr/bin/env python3
"""Real-world finance evaluation pipeline with detailed diagnostics."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import logging
import math
import random
import statistics
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - guard for environments without lightgbm
    lgb = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(SRC_ROOT))

from tools import event_study_core as esc
from tools import finance_extract as fx

try:  # pragma: no cover - optional dependency chain
    from tools.fetch_sec_real import CONTROL, DISTRESSED
except Exception:  # pragma: no cover - fallback constants
    DISTRESSED = ("CVNA", "UPST", "KSS", "CCL", "RIVN", "AA")
    CONTROL = ("AAPL", "MSFT", "JNJ", "PG", "COST", "ADBE")

DISTRESSED_SET = {ticker.upper() for ticker in DISTRESSED}
CONTROL_SET = {ticker.upper() for ticker in CONTROL}
CACHE_DIR = Path("reports/realworld/cache")
MAX_HTML_CHARS = 3_500_000


@dataclasses.dataclass
class CalibrationDetails:
    method: str
    scores: List[float]
    ece: float
    brier: float
    curve: List[Dict[str, float]]
    parameters: Dict[str, float]


@dataclasses.dataclass
class ModelEvaluation:
    name: str
    raw_scores: List[float]
    metrics: Dict[str, object]
    calibrations: Dict[str, CalibrationDetails]


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    df = df[df["path"].notna()]
    df["path"] = df["path"].astype(str)
    df = df[df["path"].map(lambda p: Path(p).exists())]
    if df.empty:
        raise RuntimeError("Manifest is empty after filtering existing filings")
    return df


def _cache_key(row: pd.Series) -> str:
    md5 = str(row.get("md5", "")).strip()
    if md5:
        return md5
    path = str(row.get("path", "")).strip()
    if path:
        return hashlib.sha1(path.encode("utf-8")).hexdigest()
    serialised = json.dumps({k: row[k] for k in row.index}, sort_keys=True)
    return hashlib.sha1(serialised.encode("utf-8")).hexdigest()


def _cache_path(row: pd.Series) -> Path:
    return CACHE_DIR / f"{_cache_key(row)}.json"


def _load_cached(row: pd.Series) -> Optional[Dict[str, object]]:
    path = _cache_path(row)
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def _save_cached(row: pd.Series, payload: Dict[str, object]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(row)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _label_for_ticker(ticker: str) -> Tuple[str, int]:
    ticker = ticker.upper()
    if ticker in DISTRESSED_SET:
        return "distressed", 1
    if ticker in CONTROL_SET:
        return "control", 0
    return "unknown", 0


def _rng_for_filing(seed: int, ticker: str, filed: str) -> random.Random:
    salt = hash((ticker.upper(), filed)) & 0xFFFFFFFF
    return random.Random(seed + salt)


def _aggregate_doc_text(doc: object) -> str:
    parts = [
        getattr(block, "text", "")
        for block in getattr(doc, "blocks", [])
        if getattr(block, "kind", None) == "paragraph"
    ]
    return "\n".join(parts)


def _evidence_to_dict(ev: esc.FilingEvidence) -> Dict[str, object]:
    return {
        "ticker": ev.ticker,
        "filing_id": ev.filing_id,
        "filing_date": ev.filing_date,
        "form": ev.form,
        "obligation": ev.obligation,
        "permission": ev.permission,
        "score": ev.score,
        "raw_score": ev.raw_score,
        "cue_weight": ev.cue_weight,
        "severity_hits": ev.severity_hits,
        "numeric_hits": ev.numeric_hits,
        "numeric_strength": ev.numeric_strength,
        "numeric_confidence": ev.numeric_confidence,
        "bypass_proximity": ev.bypass_proximity,
    }


def _evidence_from_dict(data: Dict[str, object]) -> esc.FilingEvidence:
    return esc.FilingEvidence(
        ticker=str(data.get("ticker", "")),
        filing_id=str(data.get("filing_id", "")),
        filing_date=str(data.get("filing_date", "")),
        form=str(data.get("form", "")),
        obligation=str(data.get("obligation", "")),
        permission=str(data.get("permission", "")),
        score=float(data.get("score", 0.0)),
        raw_score=float(data.get("raw_score", 0.0)),
        cue_weight=float(data.get("cue_weight", 0.0)),
        severity_hits=int(data.get("severity_hits", 0)),
        numeric_hits=int(data.get("numeric_hits", 0)),
        numeric_strength=float(data.get("numeric_strength", 0.0)),
        numeric_confidence=float(data.get("numeric_confidence", 0.0)),
        bypass_proximity=float(data.get("bypass_proximity", 0.0)),
    )


def _record_to_dict(record: esc.FilingRecord) -> Dict[str, object]:
    return {
        "ticker": record.ticker,
        "role": record.role,
        "label": record.label,
        "as_of": record.as_of.isoformat(),
        "filing_date": record.filing_date.isoformat(),
        "form": record.form,
        "filing_id": record.filing_id,
        "cce": record.cce,
        "cce_raw": record.cce_raw,
        "cue_weight": record.cue_weight,
        "severity_hits": record.severity_hits,
        "numeric_hits": record.numeric_hits,
        "numeric_strength": record.numeric_strength,
        "numeric_confidence": record.numeric_confidence,
        "bypass_proximity": record.bypass_proximity,
        "pair_count": record.pair_count,
        "evidence_rows": [_evidence_to_dict(ev) for ev in record.evidence_rows],
    }


def _record_from_dict(data: Dict[str, object]) -> esc.FilingRecord:
    return esc.FilingRecord(
        ticker=str(data.get("ticker", "")),
        role=str(data.get("role", "")),
        label=int(data.get("label", 0)),
        as_of=dt.date.fromisoformat(str(data.get("as_of", dt.date.today().isoformat()))),
        filing_date=dt.date.fromisoformat(
            str(data.get("filing_date", dt.date.today().isoformat()))
        ),
        form=str(data.get("form", "")),
        filing_id=str(data.get("filing_id", "")),
        cce=float(data.get("cce", 0.0)),
        cce_raw=float(data.get("cce_raw", 0.0)),
        cue_weight=float(data.get("cue_weight", 0.0)),
        severity_hits=int(data.get("severity_hits", 0)),
        numeric_hits=int(data.get("numeric_hits", 0)),
        numeric_strength=float(data.get("numeric_strength", 0.0)),
        numeric_confidence=float(data.get("numeric_confidence", 0.0)),
        bypass_proximity=float(data.get("bypass_proximity", 0.0)),
        pair_count=int(data.get("pair_count", 0)),
        evidence_rows=[
            _evidence_from_dict(ev)
            for ev in data.get("evidence_rows", [])
            if isinstance(ev, dict)
        ],
    )


def _serialise_processed(processed: Dict[str, object]) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "status": "ok",
        "record": _record_to_dict(processed["record"]),
        "text": processed.get("text", ""),
    }
    evidence = processed.get("evidence", [])
    payload["evidence"] = [
        _evidence_to_dict(ev) for ev in evidence if isinstance(ev, esc.FilingEvidence)
    ]
    best_pair = processed.get("best_pair")
    if best_pair is not None:
        payload["best_pair"] = {
            "obligation": getattr(best_pair, "obligation", ""),
            "permission": getattr(best_pair, "permission", ""),
            "metadata": getattr(best_pair, "metadata", {}),
        }
    return payload


def _deserialise_processed(data: Dict[str, object]) -> Dict[str, object]:
    record = _record_from_dict(data.get("record", {}))
    evidence = [
        _evidence_from_dict(ev)
        for ev in data.get("evidence", [])
        if isinstance(ev, dict)
    ]
    best_payload = data.get("best_pair") or {}
    best_pair = None
    if best_payload:
        best_pair = fx.ExtractionPair(  # type: ignore[attr-defined]
            obligation=str(best_payload.get("obligation", "")),
            permission=str(best_payload.get("permission", "")),
            metadata=dict(best_payload.get("metadata", {})),
        )
    return {
        "record": record,
        "evidence": evidence,
        "best_pair": best_pair,
        "text": str(data.get("text", "")),
    }


def _process_filing(row: pd.Series, seed: int) -> Optional[Dict[str, object]]:
    path = Path(str(row["path"]))
    html = path.read_text(encoding="utf-8", errors="ignore")
    if len(html) > MAX_HTML_CHARS:
        html = html[:MAX_HTML_CHARS]
    form = row.get("form")
    doc = fx._build_doc_from_html(html, form)  # type: ignore[attr-defined]
    pairs = fx.extract_pairs_from_doc(doc)
    if not pairs:
        return None
    filed = str(row.get("filed", ""))
    role, label = _label_for_ticker(str(row.get("ticker", "")))
    rng = _rng_for_filing(seed, str(row.get("ticker", "")), filed)

    evidence_rows: List[esc.FilingEvidence] = []
    best = {
        "score": 0.0,
        "raw": 0.0,
        "cue": 1.0,
        "sev": 0,
        "num_hits": 0,
        "num_strength": 0.0,
        "num_conf": 0.0,
        "bypass": 0.0,
        "pair": None,
    }

    for pair in pairs:
        score, raw, cue, sev, num_hits, num_strength, num_conf, bypass = esc._compute_pair_score(  # type: ignore[attr-defined]
            pair.obligation,
            pair.permission,
            rng,
        )
        evidence_rows.append(
            esc.FilingEvidence(
                ticker=str(row.get("ticker", "")).upper(),
                filing_id="unknown",
                filing_date=filed or "",
                form=form or "",
                obligation=pair.obligation.strip(),
                permission=pair.permission.strip(),
                score=score,
                raw_score=raw,
                cue_weight=cue,
                severity_hits=sev,
                numeric_hits=num_hits,
                numeric_strength=num_strength,
                numeric_confidence=num_conf,
                bypass_proximity=bypass,
            )
        )
        if score > best["score"]:
            best.update(
                {
                    "score": score,
                    "raw": raw,
                    "cue": cue,
                    "sev": sev,
                    "num_hits": num_hits,
                    "num_strength": num_strength,
                    "num_conf": num_conf,
                    "bypass": bypass,
                    "pair": pair,
                }
            )

    if not evidence_rows:
        return None

    try:
        filing_date = pd.to_datetime(filed).date()
    except Exception:
        filing_date = pd.Timestamp.utcnow().date()

    record = esc.FilingRecord(
        ticker=str(row.get("ticker", "")).upper(),
        role=role,
        label=label,
        as_of=filing_date,
        filing_date=filing_date,
        form=form or "",
        filing_id=f"{row.get('ticker','')}-{filing_date.isoformat()}-000",
        cce=float(best["score"]),
        cce_raw=float(best["raw"]),
        cue_weight=float(best["cue"]),
        severity_hits=int(best["sev"]),
        numeric_hits=int(best["num_hits"]),
        numeric_strength=float(best["num_strength"]),
        numeric_confidence=float(best["num_conf"]),
        bypass_proximity=float(best["bypass"]),
        pair_count=len(evidence_rows),
        evidence_rows=evidence_rows,
    )

    return {
        "record": record,
        "evidence": evidence_rows,
        "best_pair": best["pair"],
        "text": _aggregate_doc_text(doc),
        "md5": row.get("md5"),
        "path": str(path),
        "filed": filing_date,
        "ticker": record.ticker,
        "label": label,
        "role": role,
    }


def _feature_dataframe(records: Sequence[esc.FilingRecord]) -> Tuple[pd.DataFrame, np.ndarray]:
    X, y, names = esc._feature_matrix(records)  # type: ignore[attr-defined]
    df = pd.DataFrame(X, columns=names)
    df.drop(columns=["bias"], inplace=True, errors="ignore")
    return df, y


def _train_test_split(rows: List[Dict[str, object]]) -> Tuple[List[int], List[int]]:
    by_ticker: Dict[str, List[Tuple[int, pd.Timestamp]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        filed = row["filed"]
        by_ticker[row["ticker"]].append((idx, pd.Timestamp(filed)))
    train_indices: List[int] = []
    test_indices: List[int] = []
    for ticker, entries in by_ticker.items():
        entries.sort(key=lambda item: item[1])
        if len(entries) == 1:
            test_indices.append(entries[0][0])
            continue
        cutoff = max(1, len(entries) // 3)
        for idx, _ in entries[:-cutoff]:
            train_indices.append(idx)
        for idx, _ in entries[-cutoff:]:
            test_indices.append(idx)
    return sorted(set(train_indices)), sorted(set(test_indices))


def _bootstrap_auc(y_true: np.ndarray, scores: np.ndarray, *, seed: int, n: int = 2000) -> Tuple[Tuple[float, float], int]:
    rng = np.random.default_rng(seed)
    samples: List[float] = []
    n_obs = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_scores = scores[idx]
        samples.append(roc_auc_score(sample_y, sample_scores))
    if not samples:
        return ((math.nan, math.nan), 0)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return (float(lower), float(upper)), len(samples)


def _auc_pvalue_vs_chance(y_true: np.ndarray, scores: np.ndarray, *, seed: int, n: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    auc_obs = roc_auc_score(y_true, scores)
    count = 0
    valid = 0
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_scores = scores[idx]
        auc_boot = roc_auc_score(sample_y, sample_scores)
        valid += 1
        if auc_boot <= 0.5:
            count += 1
    if valid == 0:
        return math.nan
    if auc_obs < 0.5:
        return 1.0
    return max(count / valid, 1.0 / valid)


def _expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=bins, strategy="uniform")
    counts, _ = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    ece = 0.0
    for bin_idx in range(len(prob_true)):
        weight = counts[bin_idx] / total
        ece += weight * abs(prob_true[bin_idx] - prob_pred[bin_idx])
    return float(ece)


def _wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    phat = successes / total
    denom = 1 + z ** 2 / total
    centre = phat + z ** 2 / (2 * total)
    adj = z * math.sqrt(phat * (1 - phat) / total + z ** 2 / (4 * total ** 2))
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return float(max(0.0, lower)), float(min(1.0, upper))


def _tfidf_similarity(rows: List[Dict[str, object]], train_idx: Sequence[int], test_idx: Sequence[int]) -> Tuple[float, Tuple[int, int]]:
    texts = [rows[idx]["text"] for idx in train_idx + list(test_idx)]
    if not texts:
        return (math.nan, (-1, -1))
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    train_matrix = matrix[: len(train_idx)]
    test_matrix = matrix[len(train_idx) :]
    if train_matrix.shape[0] == 0 or test_matrix.shape[0] == 0:
        return (math.nan, (-1, -1))
    sims = cosine_similarity(test_matrix, train_matrix)
    max_pos = np.unravel_index(np.argmax(sims), sims.shape)
    max_val = float(sims[max_pos])
    test_pos, train_pos = max_pos
    return max_val, (test_idx[test_pos], train_idx[train_pos])


def _timestamp_for_row(row: Mapping[str, object]) -> pd.Timestamp:
    filed = row.get("filed")
    if isinstance(filed, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(filed)
    return pd.Timestamp(row.get("filing_date", dt.datetime.utcnow()))


def _sorted_indices_by_time(rows: Sequence[Mapping[str, object]]) -> List[int]:
    return sorted(range(len(rows)), key=lambda idx: _timestamp_for_row(rows[idx]))


def _rolling_folds(rows: Sequence[Mapping[str, object]], n_splits: int, min_train_size: int = 20) -> List[Tuple[List[int], List[int]]]:
    ordered = _sorted_indices_by_time(rows)
    folds: List[Tuple[List[int], List[int]]] = []
    total = len(ordered)
    if total < min_train_size + 1:
        return [(ordered[:max(total - 1, 1)], ordered[max(total - 1, 1):])]
    block = max(1, total // (n_splits + 1))
    for split in range(1, n_splits + 1):
        test_start = split * block
        test_end = min(total, test_start + block)
        train = ordered[:test_start]
        test = ordered[test_start:test_end]
        if len(train) < min_train_size or not test:
            continue
        folds.append((train, test))
    if not folds:
        folds.append((ordered[: total - 1], ordered[total - 1 :]))
    return folds


def _blocked_folds(rows: Sequence[Mapping[str, object]], n_splits: int) -> List[Tuple[List[int], List[int]]]:
    ordered = _sorted_indices_by_time(rows)
    total = len(ordered)
    block = max(1, total // n_splits)
    folds: List[Tuple[List[int], List[int]]] = []
    for idx in range(n_splits):
        start = idx * block
        end = total if idx == n_splits - 1 else min(total, (idx + 1) * block)
        test = ordered[start:end]
        train = ordered[:start] + ordered[end:]
        if not train or not test:
            continue
        folds.append((train, test))
    if not folds:
        mid = total // 2
        folds.append((ordered[:mid], ordered[mid:]))
    return folds


def _build_time_folds(rows: Sequence[Mapping[str, object]], strategy: str, n_splits: int, min_train_size: int = 20) -> List[Tuple[List[int], List[int]]]:
    if strategy == "rolling":
        return _rolling_folds(rows, n_splits, min_train_size=min_train_size)
    if strategy == "blocked":
        return _blocked_folds(rows, n_splits)
    raise ValueError(f"Unsupported CV strategy: {strategy}")


def _time_train_test_split(rows: Sequence[Mapping[str, object]], holdout_fraction: float) -> Tuple[List[int], List[int]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    ordered = _sorted_indices_by_time(rows)
    cutoff = max(1, int(round(len(ordered) * (1 - holdout_fraction))))
    train = ordered[:cutoff]
    test = ordered[cutoff:]
    if not test:
        test = [ordered[-1]]
        train = ordered[:-1]
    return train, test


def _remap_folds(folds: Sequence[Tuple[Sequence[int], Sequence[int]]], base_indices: Sequence[int]) -> List[Tuple[List[int], List[int]]]:
    remapped: List[Tuple[List[int], List[int]]] = []
    for train_local, test_local in folds:
        train = [base_indices[idx] for idx in train_local]
        test = [base_indices[idx] for idx in test_local]
        if train and test:
            remapped.append((train, test))
    return remapped


class _PlattCalibrator:
    def __init__(self, seed: int) -> None:
        self._model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "_PlattCalibrator":
        scores = scores.reshape(-1, 1)
        self._model.fit(scores, y)
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        scores = scores.reshape(-1, 1)
        return self._model.predict_proba(scores)[:, 1]

    @property
    def parameters(self) -> Dict[str, float]:
        coef = float(self._model.coef_.ravel()[0])
        intercept = float(self._model.intercept_.ravel()[0])
        return {"coef": coef, "intercept": intercept}


class _IsotonicCalibrator:
    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "_IsotonicCalibrator":
        self._model.fit(scores, y)
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return self._model.transform(scores)

    @property
    def parameters(self) -> Dict[str, float]:
        return {"n_inputs": len(getattr(self._model, "X_thresholds_", []))}


class _TemperatureCalibrator:
    def __init__(self) -> None:
        self._temperature = 1.0

    @staticmethod
    def _to_logit(prob: np.ndarray) -> np.ndarray:
        eps = np.finfo(float).eps
        prob = np.clip(prob, eps, 1 - eps)
        return np.log(prob / (1 - prob))

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "_TemperatureCalibrator":
        logits = self._to_logit(scores)

        def _loss(temp: float) -> float:
            scaled = logits / max(temp, 1e-3)
            prob = 1 / (1 + np.exp(-scaled))
            return -np.mean(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12))

        best_temp = 1.0
        best_loss = float("inf")
        for temp in np.linspace(0.1, 5.0, 100):
            loss = _loss(temp)
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        self._temperature = float(best_temp)
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        logits = self._to_logit(scores)
        scaled = logits / max(self._temperature, 1e-3)
        return 1 / (1 + np.exp(-scaled))

    @property
    def parameters(self) -> Dict[str, float]:
        return {"temperature": float(self._temperature)}


def _midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        midranks[i:j] = mid
        i = j
    out = np.empty(n, dtype=float)
    out[sorted_idx] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    if m == 0 or n == 0:
        return (
            np.full(predictions_sorted_transposed.shape[0], math.nan),
            np.full((predictions_sorted_transposed.shape[0], predictions_sorted_transposed.shape[0]), math.nan),
            np.full((predictions_sorted_transposed.shape[0], predictions_sorted_transposed.shape[0]), math.nan),
        )
    positives = predictions_sorted_transposed[:, :m]
    negatives = predictions_sorted_transposed[:, m:]
    tx = np.apply_along_axis(_midrank, 1, predictions_sorted_transposed)
    ty = np.apply_along_axis(_midrank, 1, positives)
    tz = np.apply_along_axis(_midrank, 1, negatives)
    aucs = (tx[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (ty - (m + 1) / 2) / n
    v10 = 1 - (tz - (n + 1) / 2) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m, sy / n


def _delong_ci(y_true: np.ndarray, scores: np.ndarray, alpha: float = 0.95) -> Tuple[float, float, float]:
    y_true = y_true.astype(int)
    order = np.argsort(-scores)
    y_true = y_true[order]
    scores = scores[order]
    label_1_count = int(y_true.sum())
    preds = scores[np.newaxis, :]
    aucs, v01, v10 = _fast_delong(preds, label_1_count)
    auc = float(aucs[0])
    var = float(v01 + v10)
    if math.isnan(var) or var <= 0:
        return auc, math.nan, math.nan
    std = math.sqrt(var)
    z = stats.norm.ppf(0.5 + alpha / 2.0)
    lower = max(0.0, auc - z * std)
    upper = min(1.0, auc + z * std)
    return auc, lower, upper


def _label_shuffle_auc(X: np.ndarray, y: np.ndarray, train_idx: Sequence[int], test_idx: Sequence[int], seed: int) -> float:
    rng = np.random.default_rng(seed)
    shuffled = y.copy()
    rng.shuffle(shuffled)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
    model.fit(X_train, shuffled[train_idx])
    scores = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y[test_idx], scores)


def _ablation_scores(
    features: pd.DataFrame,
    y: np.ndarray,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    groups: Dict[str, Sequence[str]],
    seed: int,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name, cols in groups.items():
        cols = [col for col in cols if col in features.columns]
        if not cols:
            results[name] = float("nan")
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features.iloc[train_idx][cols])
        X_test = scaler.transform(features.iloc[test_idx][cols])
        model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
        model.fit(X_train, y[train_idx])
        scores = model.predict_proba(X_test)[:, 1]
        results[name] = roc_auc_score(y[test_idx], scores)
    return results


def _threshold_sweep(
    scores: np.ndarray,
    y_true: np.ndarray,
    test_indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    ig_universe: Sequence[str],
) -> Dict[str, object]:
    ig_set = {ticker.upper() for ticker in ig_universe}
    ig_indices = []
    for idx in range(len(test_indices)):
        row = rows[test_indices[idx]]
        ticker = row.get("ticker")
        if ticker is None and "record" in row:
            ticker = getattr(row["record"], "ticker", None)
        if ticker and ticker.upper() in ig_set:
            ig_indices.append(idx)
    ig_total = sum(1 for idx in ig_indices if y_true[idx] == 0)
    if ig_total == 0:
        return {
            "threshold": 0.5,
            "ig_fp_rate": math.nan,
            "ig_fp": 0,
            "ig_total": 0,
            "wilson": (math.nan, math.nan),
        }
    best_threshold = 0.5
    best_recall = -1.0
    best_auc = -1.0
    best_payload: Dict[str, object] = {}
    thresholds = np.linspace(0.01, 0.99, 99)
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        ig_fp = int(
            sum(
                1
                for local_idx in ig_indices
                if y_true[local_idx] == 0 and preds[local_idx] == 1
            )
        )
        rate = ig_fp / ig_total if ig_total else math.nan
        wilson = _wilson_ci(ig_fp, ig_total)
        if ig_total and (rate <= 0.20) and (not math.isnan(wilson[1]) and wilson[1] <= 0.10):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average=None, labels=[0, 1], zero_division=0
            )
            recall_pos = float(recall[1])
            auc = roc_auc_score(y_true, scores)
            if recall_pos > best_recall or (math.isclose(recall_pos, best_recall) and auc > best_auc):
                best_recall = recall_pos
                best_auc = auc
                best_threshold = float(threshold)
                best_payload = {
                    "threshold": float(threshold),
                    "ig_fp_rate": float(rate),
                    "ig_fp": ig_fp,
                    "ig_total": ig_total,
                    "wilson": (float(wilson[0]), float(wilson[1])),
                    "precision": float(precision[1]),
                    "recall": recall_pos,
                    "f1": float(f1[1]),
                }
    if not best_payload:
        best_payload = {
            "threshold": best_threshold,
            "ig_fp_rate": float(0.0 if ig_total == 0 else math.nan),
            "ig_fp": 0,
            "ig_total": ig_total,
            "wilson": _wilson_ci(0, ig_total),
        }
    return best_payload


def _model_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    seed: int,
) -> Dict[str, object]:
    roc = roc_auc_score(y_true, scores)
    pr = average_precision_score(y_true, scores)
    bootstrap_ci, effective = _bootstrap_auc(y_true, scores, seed=seed)
    auc, lower, upper = _delong_ci(y_true, scores)
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], zero_division=0
    )
    ece = _expected_calibration_error(y_true, scores)
    brier = brier_score_loss(y_true, scores)
    return {
        "roc_auc": float(roc),
        "roc_auc_ci_bootstrap": tuple(float(x) for x in bootstrap_ci),
        "roc_auc_ci_delong": (float(lower), float(upper)),
        "roc_auc_delong": float(auc),
        "pr_auc": float(pr),
        "confusion_matrix": cm.tolist(),
        "precision": [float(x) for x in precision],
        "recall": [float(x) for x in recall],
        "f1": [float(x) for x in f1],
        "support": [int(x) for x in support],
        "threshold": float(threshold),
        "ece": float(ece),
        "brier": float(brier),
        "bootstrap_samples": int(effective),
    }


def _build_estimator(name: str, seed: int) -> object:
    if name == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "gbm":
        return GradientBoostingClassifier(random_state=seed)
    if name == "lightgbm":
        if lgb is None:
            raise RuntimeError("LightGBM is not installed; install lightgbm to use this model")
        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"Unknown model name: {name}")


def _predict_scores_from_estimator(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            return 1 / (1 + np.exp(-decision))
        return 1 / (1 + np.exp(-decision[:, 1]))
    preds = model.predict(X)
    return preds.astype(float)


def _cross_val_predictions(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[Sequence[int], Sequence[int]]],
    seed: int,
) -> np.ndarray:
    preds = np.full(len(y), np.nan, dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        estimator = _build_estimator(model_name, seed + fold_idx)
        estimator.fit(X[train_idx], y[train_idx])
        preds[test_idx] = _predict_scores_from_estimator(estimator, X[test_idx])
    return preds


def _calibration_curve_table(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> List[Dict[str, float]]:
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=bins, strategy="uniform")
    table: List[Dict[str, float]] = []
    for idx, (truth, pred) in enumerate(zip(prob_true, prob_pred)):
        table.append({"bin": float(idx), "prob_true": float(truth), "prob_pred": float(pred)})
    return table


def _apply_calibration(
    method: str,
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> CalibrationDetails:
    if method == "platt":
        calibrator = _PlattCalibrator(seed).fit(train_scores, train_labels)
    elif method == "isotonic":
        calibrator = _IsotonicCalibrator().fit(train_scores, train_labels)
    elif method == "temperature":
        calibrator = _TemperatureCalibrator().fit(train_scores, train_labels)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    calibrated_scores = calibrator(test_scores)
    ece = _expected_calibration_error(test_labels, calibrated_scores)
    brier = brier_score_loss(test_labels, calibrated_scores)
    curve = _calibration_curve_table(test_labels, calibrated_scores)
    return CalibrationDetails(
        method=method,
        scores=calibrated_scores.tolist(),
        ece=float(ece),
        brier=float(brier),
        curve=curve,
        parameters=getattr(calibrator, "parameters", {}),
    )


def evaluate(
    manifest_path: Path,
    *,
    seed: int = 13,
    holdout_fraction: float = 0.25,
    cv_strategy: str = "rolling",
    cv_splits: int = 5,
    models: Optional[Sequence[str]] = None,
    calibrations: Optional[Sequence[str]] = None,
    time_split: bool = True,
    export_folds: Optional[Path] = None,
) -> Dict[str, object]:
    manifest = _load_manifest(manifest_path)
    rows: List[Dict[str, object]] = []
    total = len(manifest)
    for idx, (_, row) in enumerate(manifest.iterrows(), start=1):
        ticker = str(row.get("ticker", "?"))
        filed = row.get("filed", "?")
        cached = _load_cached(row)
        if cached is not None:
            if cached.get("status") == "skip":
                logging.warning("skipped %s %s (cached)", ticker, filed)
                continue
            cached_processed = _deserialise_processed(cached)
            logging.info("cached %s/%s %s %s", idx, total, ticker, filed)
            rows.append(cached_processed)
            continue

        logging.info("extracting %s/%s %s %s", idx, total, ticker, filed)
        processed = _process_filing(row, seed)
        if processed is None:
            logging.warning("skipped %s %s", ticker, filed)
            _save_cached(row, {"status": "skip"})
            continue
        logging.info(
            "processed %s/%s %s %s",
            idx,
            total,
            processed["ticker"],
            processed["filed"],
        )
        _save_cached(row, _serialise_processed(processed))
        rows.append(processed)

    if not rows:
        return {
            "status": "no_evidence",
            "counts": {
                "total_filings": 0,
                "train": 0,
                "test": 0,
                "distressed": 0,
                "controls": 0,
            },
            "metrics": {},
            "train_indices": [],
            "test_indices": [],
            "label_shuffle_auc": math.nan,
            "welch": {"statistic": math.nan, "p_value": math.nan},
            "ablations": {},
            "similarity": {"max_cross_split": math.nan, "pair": (-1, -1)},
            "rows": [],
            "features": {},
            "fingerprint": {},
        }

    records = [row["record"] for row in rows]
    features_df, y = _feature_dataframe(records)
    X = features_df.to_numpy(dtype=float)

    if time_split:
        train_idx, test_idx = _time_train_test_split(rows, holdout_fraction)
    else:
        train_idx, test_idx = _train_test_split(rows)
    if not train_idx or not test_idx:
        raise RuntimeError("Train/test split insufficient; need filings across time")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    local_folds = _build_time_folds(
        [rows[idx] for idx in train_idx],
        cv_strategy,
        cv_splits,
        min_train_size=max(5, int(len(train_idx) * 0.3)),
    )
    global_folds = _remap_folds(local_folds, train_idx)

    if export_folds is not None:
        export_folds.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train": train_idx,
            "test": test_idx,
            "cv": [
                {"train": fold_train, "test": fold_test}
                for fold_train, fold_test in global_folds
            ],
        }
        with export_folds.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    requested_models = list(models or ("logistic", "gbm"))
    requested_calibrations = list(calibrations or ("platt", "isotonic", "temperature"))

    baseline_scores = features_df.iloc[test_idx]["CCE"].to_numpy(dtype=float)
    baseline_threshold = _threshold_sweep(baseline_scores, y_test, test_idx, rows, fx.INVESTMENT_GRADE)
    baseline_metrics = _model_metrics(y_test, baseline_scores, baseline_threshold["threshold"], seed)
    baseline_metrics["ig_guardrail"] = baseline_threshold

    model_results: Dict[str, ModelEvaluation] = {}

    for name in requested_models:
        if name == "baseline":
            model_results[name] = ModelEvaluation(
                name=name,
                raw_scores=baseline_scores.tolist(),
                metrics=baseline_metrics,
                calibrations={},
            )
            continue
        if name == "stacked":
            # Stacked ensemble will be computed after base models are available
            continue
        estimator = _build_estimator(name, seed)
        estimator.fit(X_train, y_train)
        test_scores = _predict_scores_from_estimator(estimator, X_test)

        calibration_map: Dict[str, CalibrationDetails] = {}
        if local_folds:
            oof_scores = _cross_val_predictions(name, X_train, y_train, local_folds, seed)
        else:
            oof_scores = np.full(len(y_train), np.nan)

        valid_mask = ~np.isnan(oof_scores)
        if valid_mask.sum() >= 5 and len(np.unique(y_train[valid_mask])) > 1:
            for method in requested_calibrations:
                try:
                    calibration_map[method] = _apply_calibration(
                        method,
                        oof_scores[valid_mask],
                        y_train[valid_mask],
                        test_scores,
                        y_test,
                        seed,
                    )
                except ValueError:
                    continue

        threshold = _threshold_sweep(test_scores, y_test, test_idx, rows, fx.INVESTMENT_GRADE)
        metrics = _model_metrics(y_test, test_scores, threshold["threshold"], seed)
        metrics["ig_guardrail"] = threshold
        model_results[name] = ModelEvaluation(
            name=name,
            raw_scores=test_scores.tolist(),
            metrics=metrics,
            calibrations=calibration_map,
        )

    if "stacked" in requested_models:
        base_candidates = [model_results.get(candidate) for candidate in ("logistic", "gbm", "lightgbm")]
        base_candidates = [candidate for candidate in base_candidates if candidate is not None]
        if base_candidates:
            stacked_scores = np.mean([np.array(candidate.raw_scores) for candidate in base_candidates], axis=0)
            threshold = _threshold_sweep(stacked_scores, y_test, test_idx, rows, fx.INVESTMENT_GRADE)
            metrics = _model_metrics(y_test, stacked_scores, threshold["threshold"], seed)
            metrics["ig_guardrail"] = threshold
            model_results["stacked"] = ModelEvaluation(
                name="stacked",
                raw_scores=stacked_scores.tolist(),
                metrics=metrics,
                calibrations={},
            )

    label_shuffle_auc = _label_shuffle_auc(X, y, train_idx, test_idx, seed=seed + 1)

    distress_scores = features_df.iloc[test_idx]["CCE"].to_numpy()
    distress_labels = y[test_idx]
    distressed = distress_scores[distress_labels == 1]
    controls = distress_scores[distress_labels == 0]
    welch = stats.ttest_ind(distressed, controls, equal_var=False)

    ablation_groups = {
        "sectionizer": [
            "cue_weight",
            "pair_count",
            "ev_count",
            "ev_weight_mean",
            "ev_weight_max",
        ],
        "numeric": [
            "numeric_hits",
            "numeric_strength",
            "numeric_confidence",
            "ev_numeric_hits_mean",
            "ev_numeric_hits_sum",
            "ev_numeric_strength_mean",
        ],
        "bypass": [
            "bypass_proximity",
            "ev_bypass_proximity_mean",
            "ev_bypass_proximity_max",
            "ev_bypass_proximity_min",
        ],
        "severity": [
            "severity_hits",
            "ev_severity_hits_mean",
            "ev_severity_hits_sum",
        ],
    }
    ablations = _ablation_scores(features_df, y, train_idx, test_idx, ablation_groups, seed=seed + 2)

    similarity, pair_idx = _tfidf_similarity(rows, train_idx, test_idx)

    try:
        import regex as regex_mod  # type: ignore

        regex_version = getattr(regex_mod, "__version__", "unknown")
    except Exception:  # pragma: no cover - optional dependency guard
        import re

        regex_version = getattr(re, "__version__", "stdlib")

    fingerprint_payload = {
        "regex_version": regex_version,
        "calibration_timestamp": dt.datetime.utcnow().isoformat(),
        "models": requested_models,
        "calibrations": requested_calibrations,
    }
    model_hash = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
    fingerprint_payload["model_hash"] = model_hash

    outputs = {
        "counts": {
            "total_filings": len(rows),
            "train": len(train_idx),
            "test": len(test_idx),
            "distressed": int(y.sum()),
            "controls": int(len(y) - y.sum()),
        },
        "train_indices": train_idx,
        "test_indices": test_idx,
        "cv_folds": global_folds,
        "metrics": {
            name: dataclasses.asdict(result)
            for name, result in model_results.items()
        },
        "baseline": dataclasses.asdict(
            ModelEvaluation(
                name="baseline",
                raw_scores=baseline_scores.tolist(),
                metrics=baseline_metrics,
                calibrations={},
            )
        ),
        "label_shuffle_auc": float(label_shuffle_auc),
        "welch": {
            "statistic": float(welch.statistic),
            "p_value": float(welch.pvalue),
        },
        "ablations": ablations,
        "similarity": {
            "max_cross_split": similarity,
            "pair": pair_idx,
        },
        "rows": rows,
        "features": features_df.to_dict(orient="list"),
        "fingerprint": fingerprint_payload,
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/realworld/manifest.csv"),
        help="Path to SEC manifest CSV",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=Path("reports/realworld/eval.json"))
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default="rolling",
        choices=["rolling", "blocked"],
        help="Cross-validation scheme for calibration",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--models",
        type=str,
        default="logistic,gbm",
        help="Comma-separated list of models (logistic,gbm,lightgbm,stacked,baseline)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="platt,isotonic,temperature",
        help="Comma-separated list of calibration methods",
    )
    parser.add_argument(
        "--no-time-split",
        action="store_true",
        help="Disable time-aware split and fallback to ticker-based split",
    )
    parser.add_argument(
        "--export-folds",
        type=Path,
        default=None,
        help="Optional path to export fold assignments as JSON",
    )
    args = parser.parse_args()

    model_list = [item.strip() for item in args.models.split(",") if item.strip()]
    calibration_list = [item.strip() for item in args.calibration.split(",") if item.strip()]

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    results = evaluate(
        args.manifest,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        cv_strategy=args.cv_strategy,
        cv_splits=args.cv_splits,
        models=model_list,
        calibrations=calibration_list,
        time_split=not args.no_time_split,
        export_folds=args.export_folds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(json.dumps({k: v for k, v in results.items() if k not in {"rows", "features"}}, indent=2))


if __name__ == "__main__":
    main()
