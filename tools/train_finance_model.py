#!/usr/bin/env python3
"""Train calibrated finance risk models from event-study artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Optional dependency for LightGBM users.
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    lgb = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))

from tools.event_study_core import welch_pvalue  # reuse tested helper

FEATURE_COLUMNS: Sequence[str] = (
    "CCE",
    "cce_raw",
    "cue_weight",
    "severity_hits",
    "numeric_hits",
    "numeric_strength",
    "numeric_confidence",
    "bypass_proximity",
    "pair_count",
)


@dataclass
class TrainingConfig:
    model_type: str
    param_grid: Optional[Any]
    folds: int
    seed: int


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file {path} has no rows")
    return df


def _aggregate_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(index=pd.Index([], name="filing"))
    groups = evidence.groupby("filing")

    def _series(name: str):
        if name in evidence.columns:
            return groups[name]
        return None

    summary = pd.DataFrame(index=groups.size().index)
    summary["ev_count"] = groups.size()

    for column in (
        "CCE",
        "cce_raw",
        "weight",
        "severity_hits",
        "numeric_hits",
        "numeric_strength",
        "numeric_confidence",
        "bypass_proximity",
    ):
        series = _series(column)
        if series is None:
            continue
        summary[f"ev_{column}_mean"] = series.mean()
        summary[f"ev_{column}_max"] = series.max()
        if column in {"severity_hits", "numeric_hits"}:
            summary[f"ev_{column}_sum"] = series.sum()
        if column == "bypass_proximity":
            summary[f"ev_{column}_min"] = series.min()
    return summary


def _prepare_features(event_scores: pd.DataFrame, evidence: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = event_scores.copy()
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
    evidence_summary = _aggregate_evidence(evidence)
    if not evidence_summary.empty:
        df = df.merge(
            evidence_summary,
            left_on="filing_id",
            right_index=True,
            how="left",
        )
    feature_columns = list(FEATURE_COLUMNS)
    derived_columns = [col for col in df.columns if col.startswith("ev_")]
    feature_columns.extend(sorted(derived_columns))
    df[derived_columns] = df[derived_columns].fillna(0.0)
    df[feature_columns] = df[feature_columns].astype(float).replace([np.inf, -np.inf], 0.0)
    X = df[feature_columns].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy(dtype=int)
    return X, y, feature_columns


def _build_estimator(model_type: str, seed: int) -> ClassifierMixin:
    if model_type == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=seed,
            class_weight="balanced",
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if model_type == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed; install it to use --model-type lightgbm")
        return lgb.LGBMClassifier(random_state=seed, class_weight="balanced")
    raise ValueError(f"Unsupported model type: {model_type}")


def _predict_scores(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            return 1.0 / (1.0 + np.exp(-decision))
        return 1.0 / (1.0 + np.exp(-decision[:, 1]))
    preds = model.predict(X)
    return preds.astype(float)


def _run_cv(
    estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> List[Dict[str, float]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: List[Dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        scores = _predict_scores(model, X[test_idx])
        auc = metrics.roc_auc_score(y[test_idx], scores)
        distressed = scores[y[test_idx] == 1]
        controls = scores[y[test_idx] == 0]
        p_value = welch_pvalue(distressed, controls)
        print(f"Fold {fold_idx}/{folds}: AUC={auc:.4f}, p-value={p_value:.4g}")
        results.append({"fold": float(fold_idx), "auc": float(auc), "p_value": float(p_value)})
    return results


def _evaluate_grid(
    base_estimator: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    grid: Iterable[Dict[str, Any]],
    folds: int,
    seed: int,
) -> Tuple[ClassifierMixin, Dict[str, Any], List[Dict[str, float]]]:
    best_score = -math.inf
    best_params: Dict[str, Any] = {}
    best_results: List[Dict[str, float]] = []
    best_model: Optional[ClassifierMixin] = None

    for params in grid:
        estimator = clone(base_estimator)
        estimator.set_params(**params)
        print(f"Evaluating params: {json.dumps(params, sort_keys=True)}")
        cv_results = _run_cv(estimator, X, y, folds=folds, seed=seed)
        mean_auc = float(np.mean([row["auc"] for row in cv_results]))
        std_auc = float(np.std([row["auc"] for row in cv_results]))
        print(f"  -> mean AUC={mean_auc:.4f} ± {std_auc:.4f}")
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = dict(params)
            best_results = cv_results
            best_model = estimator
    if best_model is None:
        raise RuntimeError("Cross-validation failed to evaluate any parameter sets")
    best_model.set_params(**best_params)
    return best_model, best_params, best_results


def _train_final_model(model: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> ClassifierMixin:
    model.fit(X, y)
    return model


def _summarise_scores(model: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    scores = _predict_scores(model, X)
    auc = metrics.roc_auc_score(y, scores)
    p_value = welch_pvalue(scores[y == 1], scores[y == 0])
    return {"auc": float(auc), "p_value": float(p_value)}


def _prepare_param_grid(raw: Optional[str]) -> Optional[Any]:
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, (dict, list)):
        raise ValueError("Parameter grid must be a mapping (or list of mappings) of parameter names to lists")
    return payload


def _ensure_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_artifacts(
    model: ClassifierMixin,
    model_dir: Path,
    feature_names: Sequence[str],
    cv_results: Sequence[Dict[str, float]],
    best_params: Dict[str, Any],
    train_metrics: Dict[str, float],
    X: np.ndarray,
) -> None:
    import joblib  # local import to keep optional dependency contained

    _ensure_model_dir(model_dir)
    model_path = model_dir / "finance_model.joblib"
    joblib.dump(model, model_path)

    feature_stats = {
        "feature_names": list(feature_names),
        "mean": list(map(float, np.mean(X, axis=0))),
        "std": list(map(float, np.std(X, axis=0))),
    }

    metadata = {
        "model_path": model_path.name,
        "feature_names": list(feature_names),
        "best_params": best_params,
        "cv_results": cv_results,
        "train_metrics": train_metrics,
        "feature_stats": feature_stats,
        "model_type": type(model).__name__,
    }
    meta_path = model_dir / "model_metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


def train_finance_model(args: argparse.Namespace) -> None:
    event_scores = _load_dataframe(Path(args.event_scores))
    evidence = _load_dataframe(Path(args.evidence))
    X, y, feature_names = _prepare_features(event_scores, evidence)
    config = TrainingConfig(
        model_type=args.model_type,
        param_grid=args.param_grid,
        folds=args.folds,
        seed=args.seed,
    )
    base_estimator = _build_estimator(config.model_type, config.seed)

    if config.param_grid:
        grid = ParameterGrid(config.param_grid)
        estimator, best_params, cv_results = _evaluate_grid(
            base_estimator,
            X,
            y,
            grid,
            config.folds,
            config.seed,
        )
    else:
        estimator = base_estimator
        cv_results = _run_cv(estimator, X, y, folds=config.folds, seed=config.seed)
        best_params = {}

    final_model = _train_final_model(estimator, X, y)
    train_metrics = _summarise_scores(final_model, X, y)
    print(f"Training AUC={train_metrics['auc']:.4f}, p-value={train_metrics['p_value']:.4g}")

    _save_artifacts(
        final_model,
        Path(args.model_dir),
        feature_names,
        cv_results,
        best_params,
        train_metrics,
        X,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-scores",
        default="reports/leverage_alpha/event_scores.csv",
        help="Path to the per-filing manifest CSV produced by the event study.",
    )
    parser.add_argument(
        "--evidence",
        default="reports/leverage_alpha/evidence.csv",
        help="Path to the per-evidence CSV produced by the event study.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/finance",
        help="Directory where the trained model and metadata should be saved.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest", "gradient_boosting", "lightgbm"],
        default="logistic",
        help="Estimator family to train.",
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        help="JSON mapping (or path to JSON) specifying hyperparameter sweeps.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.param_grid:
        args.param_grid = _prepare_param_grid(args.param_grid)
    train_finance_model(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
