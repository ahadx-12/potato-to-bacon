#!/usr/bin/env python3
"""Real-world finance evaluation pipeline with detailed diagnostics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:  # pragma: no cover - ensure local package import
    sys.path.insert(0, str(SRC_DIR))

from tools import event_study_core as esc
from tools import finance_extract as fx
try:  # pragma: no cover - optional dependency for fetching SEC manifests
    from tools.fetch_sec_real import CONTROL, DISTRESSED
except ModuleNotFoundError:  # pragma: no cover - fallback when downloader not installed
    print("[warn] sec_edgar_downloader not available; defaulting CONTROL/DISTRESSED lists to empty", file=sys.stderr)
    CONTROL = []
    DISTRESSED = []
from tools.finance_dedup import (
    max_cross_minhash_similarity,
    max_cross_tfidf_similarity,
)

DISTRESSED_SET = {ticker.upper() for ticker in DISTRESSED}
CONTROL_SET = {ticker.upper() for ticker in CONTROL}
CACHE_DIR = Path("reports/realworld/cache")
MAX_HTML_CHARS = 3_500_000


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    df = df[df["path"].notna()]
    df["path"] = df["path"].astype(str)
    df = df[df["path"].map(lambda p: Path(p).exists())]
    if df.empty:
        print("[warn] evaluation manifest is empty after filtering existing filings", file=sys.stderr)
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


def _row_year(row: Dict[str, object]) -> Optional[int]:
    filed = row.get("filed")
    if filed is None:
        return None
    if isinstance(filed, (dt.datetime, dt.date)):
        return filed.year
    try:
        ts = pd.to_datetime(filed)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int(ts.year)


def _issuer_split_stats(
    rows: List[Dict[str, object]], train_idx: Sequence[int], test_idx: Sequence[int]
) -> Tuple[float, List[str], List[str], List[str]]:
    train_tickers = {str(rows[idx].get("ticker", "")).upper() for idx in train_idx if rows[idx].get("ticker")}
    test_tickers = {str(rows[idx].get("ticker", "")).upper() for idx in test_idx if rows[idx].get("ticker")}
    exclusive = sorted(ticker for ticker in test_tickers if ticker not in train_tickers)
    ratio = len(exclusive) / len(test_tickers) if test_tickers else 0.0
    return ratio, exclusive, sorted(test_tickers), sorted(train_tickers)


def _strict_time_split(
    rows: List[Dict[str, object]],
    *,
    min_test: int = 30,
    issuer_ratio: float = 0.2,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    years = sorted({year for row in rows if (year := _row_year(row)) is not None})
    if len(years) < 2:
        raise RuntimeError("Temporal split requires filings across at least two years")

    def split_for_year(cutoff_year: int) -> Tuple[List[int], List[int]]:
        train: List[int] = []
        test: List[int] = []
        for idx, row in enumerate(rows):
            year = _row_year(row)
            if year is None:
                train.append(idx)
                continue
            if year <= cutoff_year:
                train.append(idx)
            else:
                test.append(idx)
        return train, test

    initial_cutoff: Optional[int] = None
    for candidate in years[:-1]:
        train_idx, test_idx = split_for_year(candidate)
        if len(test_idx) >= min_test:
            initial_cutoff = candidate
    if initial_cutoff is None:
        initial_cutoff = years[-2]
    train_idx, test_idx = split_for_year(initial_cutoff)
    if len(test_idx) < min_test:
        raise RuntimeError("Unable to satisfy minimum test size with temporal split")

    ratio, exclusive, test_issuers, train_issuers = _issuer_split_stats(rows, train_idx, test_idx)
    cutoff_year = initial_cutoff
    adjustments: List[Dict[str, object]] = []
    current_index = years.index(initial_cutoff)
    while ratio < issuer_ratio and current_index < len(years) - 1:
        next_cutoff = years[current_index + 1]
        candidate_train, candidate_test = split_for_year(next_cutoff)
        if len(candidate_test) < min_test:
            adjustments.append(
                {
                    "cutoff_year": next_cutoff,
                    "action": "halt",
                    "reason": "min_test",
                    "test_size": len(candidate_test),
                }
            )
            break
        cutoff_year = next_cutoff
        train_idx, test_idx = candidate_train, candidate_test
        ratio, exclusive, test_issuers, train_issuers = _issuer_split_stats(rows, train_idx, test_idx)
        adjustments.append(
            {
                "cutoff_year": cutoff_year,
                "issuer_ratio": ratio,
                "test_size": len(test_idx),
            }
        )
        current_index += 1

    print(
        f"[split] cutoff_year={cutoff_year} test_size={len(test_idx)} issuer_ratio={ratio:.1%} exclusive_test={len(exclusive)}",
        flush=True,
    )

    metadata = {
        "strategy": "temporal_cutoff",
        "initial_cutoff_year": initial_cutoff,
        "cutoff_year": cutoff_year,
        "min_test": min_test,
        "issuer_ratio_target": issuer_ratio,
        "issuer_ratio": ratio,
        "exclusive_test_issuers": exclusive,
        "test_issuers": test_issuers,
        "train_issuers": train_issuers,
        "adjustments": adjustments,
    }
    return sorted(set(train_idx)), sorted(set(test_idx)), metadata


def _legacy_split(rows: List[Dict[str, object]]) -> Tuple[List[int], List[int]]:
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


def _train_test_split(
    rows: List[Dict[str, object]],
    *,
    time_split: bool,
    min_test: int = 30,
    issuer_ratio: float = 0.2,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    if time_split:
        return _strict_time_split(rows, min_test=min_test, issuer_ratio=issuer_ratio)
    train_idx, test_idx = _legacy_split(rows)
    metadata = {
        "strategy": "ticker_last_third",
        "cutoff_year": None,
        "issuer_ratio": None,
        "exclusive_test_issuers": [],
    }
    return train_idx, test_idx, metadata


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


def evaluate(
    manifest_path: Path,
    seed: int = 13,
    *,
    time_split: bool = False,
    quarantine_path: Optional[Path] = None,
    decisions_path: Optional[Path] = None,
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
                print(f"[warn] skipped {ticker} {filed} (cached)", file=sys.stderr)
                continue
            cached_processed = _deserialise_processed(cached)
            print(
                f"[info] cached {idx}/{total} {ticker} {filed}",
                flush=True,
            )
            rows.append(cached_processed)
            continue

        print(f"[info] extracting {idx}/{total} {ticker} {filed}", flush=True)
        processed = _process_filing(row, seed)
        if processed is None:
            print(f"[warn] skipped {ticker} {filed}", file=sys.stderr)
            _save_cached(row, {"status": "skip"})
            continue
        print(
            f"[info] processed {idx}/{total} {processed['ticker']} {processed['filed']}",
            flush=True,
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
                "train_breakdown": {"total": 0, "by_role": {}, "by_issuer": {}, "by_year": {}},
                "test_breakdown": {"total": 0, "by_role": {}, "by_issuer": {}, "by_year": {}},
            },
            "metrics": {},
            "train_indices": [],
            "test_indices": [],
            "label_shuffle": {"auc": math.nan, "seed": None},
            "welch": {"statistic": math.nan, "p_value": math.nan},
            "ig_guardrail": {"rate": math.nan, "count": 0, "total": 0, "ci": [math.nan, math.nan]},
            "ablations": {},
            "similarity": {
                "tfidf": {"max_cross_split": math.nan, "pair": (-1, -1)},
                "minhash": {"max_cross_split": math.nan, "pair": (-1, -1)},
            },
            "split": {"strategy": "temporal_cutoff", "cutoff_year": None},
            "test_filings_after_filtering": 0,
            "dedup": {"quarantined": 0, "decisions": []},
            "rows": [],
            "features": {},
        }

    records = [row["record"] for row in rows]
    features_df, y = _feature_dataframe(records)
    X = features_df.to_numpy(dtype=float)
    train_idx, test_idx, split_metadata = _train_test_split(rows, time_split=time_split)
    if not train_idx or not test_idx:
        raise RuntimeError("Train/test split insufficient; need filings across time")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    logistic = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
    logistic.fit(X_train, y[train_idx])
    logistic_scores = logistic.predict_proba(X_test)[:, 1]

    gbr = GradientBoostingClassifier(random_state=seed)
    gbr.fit(X[train_idx], y[train_idx])
    gbr_scores = gbr.predict_proba(X[test_idx])[:, 1]

    baseline_scores = X[test_idx, features_df.columns.get_loc("CCE")]

    def _metrics(scores: np.ndarray) -> Dict[str, object]:
        roc = roc_auc_score(y[test_idx], scores)
        pr = average_precision_score(y[test_idx], scores)
        auc_ci, effective = _bootstrap_auc(y[test_idx], scores, seed=seed)
        p_value = _auc_pvalue_vs_chance(y[test_idx], scores, seed=seed)
        preds = (scores >= 0.5).astype(int)
        cm = confusion_matrix(y[test_idx], preds, labels=[0, 1])
        precision, recall, f1, support = precision_recall_fscore_support(
            y[test_idx], preds, labels=[0, 1], zero_division=0
        )
        ece = _expected_calibration_error(y[test_idx], scores)
        brier = brier_score_loss(y[test_idx], scores)
        return {
            "roc_auc": roc,
            "roc_auc_ci": auc_ci,
            "pr_auc": pr,
            "p_value_vs_chance": p_value,
            "bootstrap_samples": effective,
            "confusion_matrix": cm.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
            "ece": ece,
            "brier": brier,
        }

    metrics = {
        "baseline": _metrics(baseline_scores),
        "logistic": _metrics(logistic_scores),
        "gradient_boosting": _metrics(gbr_scores),
    }

    label_shuffle_seed = seed + 1
    label_shuffle_auc = _label_shuffle_auc(X, y, train_idx, test_idx, seed=label_shuffle_seed)

    distress_scores = features_df.iloc[test_idx]["CCE"].to_numpy()
    distress_labels = y[test_idx]
    distressed = distress_scores[distress_labels == 1]
    controls = distress_scores[distress_labels == 0]
    welch = stats.ttest_ind(distressed, controls, equal_var=False)

    ig_mask = [idx for idx in test_idx if rows[idx]["ticker"] in fx.INVESTMENT_GRADE]
    ig_fp = 0
    ig_total = 0
    if ig_mask:
        ig_total = sum(1 for idx in ig_mask if y[idx] == 0)
        ig_preds = (logistic_scores[[test_idx.index(idx) for idx in ig_mask]] >= 0.5).astype(int)
        ig_fp = int((ig_preds == 1).sum())
    ig_rate = ig_fp / ig_total if ig_total else math.nan
    ig_ci = _wilson_ci(ig_fp, ig_total) if ig_total else (math.nan, math.nan)

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

    train_texts = [rows[idx]["text"] for idx in train_idx]
    test_texts = [rows[idx]["text"] for idx in test_idx]
    tfidf_similarity, tfidf_pair_local = max_cross_tfidf_similarity(train_texts, test_texts)
    if not math.isnan(tfidf_similarity) and tfidf_pair_local != (-1, -1):
        tfidf_pair = (train_idx[tfidf_pair_local[0]], test_idx[tfidf_pair_local[1]])
    else:
        tfidf_pair = (-1, -1)

    minhash_similarity, minhash_pair_local = max_cross_minhash_similarity(train_texts, test_texts)
    if not math.isnan(minhash_similarity) and minhash_pair_local != (-1, -1):
        minhash_pair = (train_idx[minhash_pair_local[0]], test_idx[minhash_pair_local[1]])
    else:
        minhash_pair = (-1, -1)

    def _split_counts(indices: Sequence[int]) -> Dict[str, object]:
        subset = [rows[idx] for idx in indices]
        frame = pd.DataFrame(
            [
                {
                    "ticker": entry.get("ticker", "unknown"),
                    "role": entry.get("role", "unknown"),
                    "year": _row_year(entry),
                }
                for entry in subset
            ]
        )
        if frame.empty:
            return {
                "total": 0,
                "by_role": {},
                "by_issuer": {},
                "by_year": {},
            }
        return {
            "total": int(len(indices)),
            "by_role": {str(k): int(v) for k, v in frame["role"].value_counts().to_dict().items()},
            "by_issuer": {str(k): int(v) for k, v in frame["ticker"].value_counts().to_dict().items()},
            "by_year": {str(k): int(v) for k, v in frame["year"].value_counts().to_dict().items()},
        }

    outputs = {
        "counts": {
            "total_filings": len(rows),
            "train": len(train_idx),
            "test": len(test_idx),
            "distressed": int(y.sum()),
            "controls": int(len(y) - y.sum()),
            "train_breakdown": _split_counts(train_idx),
            "test_breakdown": _split_counts(test_idx),
        },
        "metrics": metrics,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "label_shuffle": {
            "auc": label_shuffle_auc,
            "seed": label_shuffle_seed,
        },
        "welch": {
            "statistic": float(welch.statistic),
            "p_value": float(welch.pvalue),
        },
        "ig_guardrail": {
            "rate": ig_rate,
            "count": ig_fp,
            "total": ig_total,
            "ci": ig_ci,
        },
        "ablations": ablations,
        "similarity": {
            "tfidf": {
                "max_cross_split": tfidf_similarity,
                "pair": tfidf_pair,
            },
            "minhash": {
                "max_cross_split": minhash_similarity,
                "pair": minhash_pair,
            },
        },
        "split": split_metadata,
        "test_filings_after_filtering": len(test_idx),
        "rows": rows,
        "features": features_df.to_dict(orient="list"),
    }

    dedup_info: Dict[str, object] = {"quarantined": 0, "decisions": []}
    if decisions_path and decisions_path.exists():
        try:
            decisions_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
            if isinstance(decisions_payload, list):
                dedup_info["decisions"] = decisions_payload
        except Exception as exc:  # pragma: no cover - best effort to continue
            print(f"[warn] failed to read dedup decisions: {exc}", file=sys.stderr)
    if quarantine_path and quarantine_path.exists():
        quarantine_df = pd.read_csv(quarantine_path)
        dedup_info["quarantined"] = int(len(quarantine_df))
        if not dedup_info["decisions"] and {
            "duplicate_of",
            "dedup_method",
            "dedup_score",
        }.issubset(set(quarantine_df.columns)):
            dedup_info["decisions"] = [
                {
                    "kept_id": row.get("duplicate_of"),
                    "removed_id": row.get("filing_id") or row.get("path"),
                    "kept_path": row.get("duplicate_path"),
                    "removed_path": row.get("path"),
                    "method": row.get("dedup_method"),
                    "score": row.get("dedup_score"),
                    "rationale": row.get("dedup_rationale"),
                }
                for _, row in quarantine_df.iterrows()
            ]
    outputs["dedup"] = dedup_info
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
    parser.add_argument(
        "--time-split",
        action="store_true",
        help="Use strict temporal cutoff when forming train/test splits",
    )
    parser.add_argument("--quarantine", type=Path, default=None, help="Dedup quarantine CSV")
    parser.add_argument(
        "--decisions-json",
        type=Path,
        default=None,
        help="Dedup decisions JSON emitted by finance_dedup",
    )
    parser.add_argument("--report", type=Path, default=None, help="Output report JSON path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Deprecated alias for --report",
    )
    args = parser.parse_args()

    report_path = args.report or args.output or Path("reports/realworld/phase1_report.json")

    quarantine_path = args.quarantine
    if quarantine_path is None:
        stem = args.manifest.stem
        suffix = args.manifest.suffix
        if stem.endswith("_dedup"):
            candidate = args.manifest.with_name(stem.replace("_dedup", "_quarantine") + suffix)
            if candidate.exists():
                quarantine_path = candidate

    results = evaluate(
        args.manifest,
        seed=args.seed,
        time_split=args.time_split,
        quarantine_path=quarantine_path,
        decisions_path=args.decisions_json,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    summary_payload = {
        k: v for k, v in results.items() if k not in {"rows", "features"}
    }
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
