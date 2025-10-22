"""Conflict Coherence Score (CCS) computation for CALE."""

from __future__ import annotations

from typing import Dict

import numpy as np

try:  # pragma: no cover - torch is optional in CI
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback to numpy backend
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .embed import D_S, JURISDICTIONS
from .types import ConflictAnalysis, LegalRule

DEFAULT_WEIGHTS = {"w_CI": 2.0, "w_K": 1.5, "w_H": 1.0, "w_TD": 0.8}


def _block_diagonal_scales(di_s: float, di_i: float, di_t: float, di_j: float, *, di: int, dj: int) -> np.ndarray:
    return np.concatenate(
        [
            np.full(D_S, di_s, dtype=np.float32),
            np.full(di, di_i, dtype=np.float32),
            np.array([di_t], dtype=np.float32),
            np.full(dj, di_j, dtype=np.float32),
        ],
        dtype=np.float32,
    )


def _to_backend_array(value: np.ndarray | float) -> np.ndarray | "torch.Tensor":
    if _TORCH_AVAILABLE:
        return torch.as_tensor(value, dtype=torch.float32)  # type: ignore[return-value]
    return np.asarray(value, dtype=np.float32)


def _backend_exp(value):
    return torch.exp(value) if _TORCH_AVAILABLE else np.exp(value)


def _backend_sum(value):
    return torch.sum(value) if _TORCH_AVAILABLE else np.sum(value)


def _backend_sigmoid(value: float) -> float:
    if _TORCH_AVAILABLE:
        return float(torch.sigmoid(torch.as_tensor(value, dtype=torch.float32)))
    return float(1.0 / (1.0 + np.exp(-value)))


SIGMA_SCALES = {
    "textualist": (0.2, 1.0, 1.2, 0.8),
    "living": (0.9, 0.5, 0.4, 0.3),
    "pragmatic": (0.7, 0.7, 0.5, 0.6),
}


class CCSCalculator:
    """Compute CCS scores under multiple interpretive philosophies."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or DEFAULT_WEIGHTS

    @staticmethod
    def compute_temporal_drift(rule1: LegalRule, rule2: LegalRule, lambda_: float = 0.1) -> float:
        t1 = getattr(rule1, "enactment_year", 1900)
        t2 = getattr(rule2, "enactment_year", 1900)
        return float(1.0 - np.exp(-lambda_ * abs(int(t1) - int(t2)) / 100.0))

    @staticmethod
    def _kernel_rbf(x1, x2, inv_diag):
        diff = x1 - x2
        exponent = -0.5 * _backend_sum(diff * diff * inv_diag)
        return _backend_exp(exponent)

    def _inv_diag_for(self, rule1: LegalRule, rule2: LegalRule, philosophy: str):
        di = len(rule1.interpretive_vec)  # type: ignore[arg-type]
        dj = len(JURISDICTIONS)
        scales = SIGMA_SCALES[philosophy]
        diag = _block_diagonal_scales(*scales, di=di, dj=dj)
        backend_diag = _to_backend_array(diag)
        return 1.0 / (backend_diag + 1e-6)

    def _inv_diag_for_dummy(self, d_total: int, philosophy: str):
        dj = len(JURISDICTIONS)
        di = d_total - D_S - 1 - dj
        scales = SIGMA_SCALES[philosophy]
        diag = _block_diagonal_scales(*scales, di=di, dj=dj)
        backend_diag = _to_backend_array(diag)
        return 1.0 / (backend_diag + 1e-6)

    def compute_kernel(self, x1: np.ndarray, x2: np.ndarray, philosophy: str) -> float:
        arr1 = _to_backend_array(x1.astype(np.float32))
        arr2 = _to_backend_array(x2.astype(np.float32))
        inv_diag = self._inv_diag_for_dummy(len(x1), philosophy)
        value = self._kernel_rbf(arr1, arr2, inv_diag)
        if _TORCH_AVAILABLE:
            return float(value.item())
        return float(value)

    def compute_ccs(self, rule1: LegalRule, rule2: LegalRule, CI: float, philosophy: str) -> float:
        x1 = _to_backend_array(rule1.feature_vector.astype(np.float32))
        x2 = _to_backend_array(rule2.feature_vector.astype(np.float32))
        inv_diag = self._inv_diag_for(rule1, rule2, philosophy)
        kernel_value = self._kernel_rbf(x1, x2, inv_diag)
        kernel_float = float(kernel_value.item()) if _TORCH_AVAILABLE else float(kernel_value)
        authority = float(min(rule1.authority_score, rule2.authority_score))
        temporal_drift = float(self.compute_temporal_drift(rule1, rule2))
        z = (
            self.weights["w_CI"] * float(CI)
            + self.weights["w_K"] * kernel_float
            + self.weights["w_H"] * authority
            - self.weights["w_TD"] * temporal_drift
        )
        return _backend_sigmoid(float(z))

    def compute_multiperspective(self, rule1: LegalRule, rule2: LegalRule, CI: float) -> ConflictAnalysis:
        scores = {p: self.compute_ccs(rule1, rule2, CI, p) for p in ("textualist", "living", "pragmatic")}
        return ConflictAnalysis(
            rule1=rule1,
            rule2=rule2,
            CI=float(CI),
            K=float(self.compute_kernel(rule1.feature_vector, rule2.feature_vector, "pragmatic")),
            H=float(min(rule1.authority_score, rule2.authority_score)),
            TD=float(self.compute_temporal_drift(rule1, rule2)),
            CCS_textualist=float(scores["textualist"]),
            CCS_living=float(scores["living"]),
            CCS_pragmatic=float(scores["pragmatic"]),
        )


__all__ = ["CCSCalculator", "DEFAULT_WEIGHTS", "SIGMA_SCALES"]
