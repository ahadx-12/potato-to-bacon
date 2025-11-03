"""Calibration helper for CALE conflict weights and thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.potatobacon.cale.engine import (  # type: ignore[attr-defined]
    CALEEngine,
    DEFAULT_WEIGHTS,
    FINANCE_CONFIG_PATH,
    _modality_scalar,
    detects_bypass,
)


@dataclass
class CalibratedParams:
    alpha: float
    beta: float
    eta: float
    threshold: float
    rmse: float


def _load_labels(path: Path) -> List[Tuple[str, str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            (row["rule1_id"], row["rule2_id"], float(row["y_expert"]))
            for row in reader
        ]


def _feature_vector(engine: CALEEngine, rule1_id: str, rule2_id: str) -> Tuple[np.ndarray, float]:
    if not engine.services:
        raise RuntimeError("CALE services not initialised")
    corpus = {rule.id: rule for rule in engine.services.corpus}
    try:
        rule1 = corpus[rule1_id]
        rule2 = corpus[rule2_id]
    except KeyError as exc:
        raise KeyError(f"Rule {exc.args[0]!r} not found in corpus") from exc
    (
        _symbolic,
        conflict_intensity,
        semantic_overlap,
        _temporal_drift,
        _authority_balance,
    ) = engine._compute_conflict_metrics(rule1, rule2)
    modality_gap = abs(_modality_scalar(rule1.text) - _modality_scalar(rule2.text))
    bypass_flag = max(
        detects_bypass(rule1.text, rule2.text),
        detects_bypass(rule2.text, rule1.text),
    )
    features = np.array(
        [1.0 - float(semantic_overlap), float(modality_gap), float(bypass_flag)],
        dtype=np.float32,
    )
    return features, float(conflict_intensity)


def calibrate(engine: CALEEngine, labels: Iterable[Tuple[str, str, float]]) -> CalibratedParams:
    X: List[np.ndarray] = []
    y: List[float] = []
    for rule1_id, rule2_id, target in labels:
        features, _ = _feature_vector(engine, rule1_id, rule2_id)
        X.append(features)
        y.append(target)
    if not X:
        raise ValueError("No calibration samples provided")
    matrix = np.vstack(X)
    targets = np.asarray(y, dtype=np.float32)
    ridge = np.eye(matrix.shape[1], dtype=np.float32) * 1e-6
    coeffs = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ targets)
    coeffs = np.maximum(coeffs, 0.0)
    if coeffs.sum() > 0:
        coeffs = coeffs / coeffs.sum()
    else:
        coeffs = np.array(
            [DEFAULT_WEIGHTS["alpha"], DEFAULT_WEIGHTS["beta"], DEFAULT_WEIGHTS["eta"]],
            dtype=np.float32,
        )
        coeffs = coeffs / coeffs.sum()
    predictions = matrix @ coeffs
    threshold = float(np.median(predictions))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    return CalibratedParams(
        alpha=float(coeffs[0]),
        beta=float(coeffs[1]),
        eta=float(coeffs[2]),
        threshold=threshold,
        rmse=rmse,
    )


def _update_config(params: CalibratedParams, *, config_path: Path) -> None:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyyaml is required to write calibration output") from exc
    if not config_path.exists():
        raise FileNotFoundError(f"Finance configuration missing at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    weights = data.setdefault("weights", {})
    weights["alpha"] = float(round(params.alpha, 4))
    weights["beta"] = float(round(params.beta, 4))
    weights["eta"] = float(round(params.eta, 4))
    signals = data.setdefault("signals", {})
    signals.setdefault("min_semantic_overlap", DEFAULT_WEIGHTS["min_semantic_overlap"])
    signals["calibration_threshold"] = float(round(params.threshold, 4))
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        default=Path("data/cale/expert_labels.csv"),
        type=Path,
        help="CSV file containing rule pair labels.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Persist calibrated parameters back into configs/finance.yml.",
    )
    parser.add_argument(
        "--config",
        default=FINANCE_CONFIG_PATH,
        type=Path,
        help="Target configuration file to update when --write-config is set.",
    )
    args = parser.parse_args()

    engine = CALEEngine()
    labels = _load_labels(args.labels)
    params = calibrate(engine, labels)

    print("Calibrated weights:")
    print(f"  alpha = {params.alpha:.4f}")
    print(f"  beta  = {params.beta:.4f}")
    print(f"  eta   = {params.eta:.4f}")
    print(f"Recommended conflict threshold: {params.threshold:.4f}")
    print(f"RMSE versus expert labels: {params.rmse:.4f}")

    if args.write_config:
        _update_config(params, config_path=args.config)
        print(f"Updated configuration written to {args.config}")


if __name__ == "__main__":
    main()
