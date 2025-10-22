"""Training utilities for the CALE conflict model."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch optional in CI
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[no-redef]
        """Minimal stub to satisfy type checking when torch is unavailable."""

        def __len__(self) -> int:  # pragma: no cover - stub
            raise NotImplementedError

        def __getitem__(self, index: int):  # pragma: no cover - stub
            raise NotImplementedError

    class _ModuleBase:  # pragma: no cover - stub
        pass

    class nn:  # type: ignore[no-redef]
        Module = _ModuleBase

    def clip_grad_norm_(*_args, **_kwargs):  # pragma: no cover - stub
        return 0.0
else:
    clip_grad_norm_ = torch.nn.utils.clip_grad_norm_

from .ccs import CCSCalculator, DEFAULT_WEIGHTS
from .symbolic import SymbolicConflictChecker
from .types import LegalRule

_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
if TORCH_AVAILABLE:  # pragma: no branch
    torch.manual_seed(_SEED)


class LegalConflictDataset(Dataset):
    """Dataset of rule pairs annotated with conflict intensities."""

    def __init__(
        self,
        *,
        pairs: Optional[Sequence[Tuple[LegalRule, LegalRule, float]]] = None,
        csv_path: Optional[Path | str] = None,
        corpus: Optional[Sequence[LegalRule]] = None,
        symbolic: Optional[SymbolicConflictChecker] = None,
    ) -> None:
        if pairs is None and csv_path is None:
            raise ValueError("Provide either labelled pairs or a CSV path")

        self._items: List[Tuple[LegalRule, LegalRule, float, float]] = []
        self._symbolic = symbolic

        if pairs is not None:
            for rule1, rule2, label in pairs:
                ci = self._compute_ci(rule1, rule2)
                self._items.append((rule1, rule2, float(label), ci))

        if csv_path is not None:
            if corpus is None:
                raise ValueError("Loading from CSV requires a corpus of rules")
            mapping = {rule.id: rule for rule in corpus}
            with open(Path(csv_path), "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rid1 = row.get("rule1_id")
                    rid2 = row.get("rule2_id")
                    label = row.get("y_expert")
                    if not rid1 or not rid2 or label is None:
                        continue
                    rule1 = mapping.get(rid1)
                    rule2 = mapping.get(rid2)
                    if rule1 is None or rule2 is None:
                        continue
                    ci = self._compute_ci(rule1, rule2)
                    self._items.append((rule1, rule2, float(label), ci))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __getitem__(self, index: int) -> Tuple[LegalRule, LegalRule, float, float]:
        return self._items[index]

    def _compute_ci(self, rule1: LegalRule, rule2: LegalRule) -> float:
        if self._symbolic is None:
            return 0.0
        try:
            return float(self._symbolic.check_conflict(rule1, rule2))
        except Exception:
            return 0.0


class CALELoss:
    """Bundle of loss components used by the trainer."""

    @staticmethod
    def compute_supervised_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for CALELoss operations")
        return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target)

    @staticmethod
    def compute_ssl_loss(
        pred: torch.Tensor, pseudo: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for CALELoss operations")
        weight = confidence.clamp(0.0, 1.0)
        loss = F.binary_cross_entropy(pred, pseudo.detach(), reduction="none")
        return torch.mean(weight * loss)

    @staticmethod
    def compute_l1_loss(weights: torch.Tensor, sigma_diag: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for CALELoss operations")
        return torch.sum(weights.abs()) + torch.sum(sigma_diag.abs())

    @staticmethod
    def compute_graph_loss(ccs_vector: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for CALELoss operations")
        vec = ccs_vector.view(1, -1)
        lap = laplacian
        if lap.device != vec.device:
            lap = lap.to(vec.device)
        result = vec @ lap @ vec.t()
        return result.squeeze()


@dataclass
class _TrainingExample:
    rule1: LegalRule
    rule2: LegalRule
    label: float
    ci: float


class CALETrainer(nn.Module):
    """End-to-end trainer for the CALE model."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required to instantiate CALETrainer")

        torch.manual_seed(_SEED)
        np.random.seed(_SEED)
        random.seed(_SEED)

        self.feature_dim = feature_dim
        weights = torch.tensor(
            [
                DEFAULT_WEIGHTS["w_CI"],
                DEFAULT_WEIGHTS["w_K"],
                DEFAULT_WEIGHTS["w_H"],
                DEFAULT_WEIGHTS["w_TD"],
            ],
            dtype=torch.float32,
        )
        self.weights = nn.Parameter(weights)
        self.sigma_diag = nn.Parameter(torch.full((feature_dim,), 0.7, dtype=torch.float32))
        self.loss = CALELoss()

    def forward(
        self, rule1: LegalRule, rule2: LegalRule, ci_value: torch.Tensor | float
    ) -> torch.Tensor:
        device = self.weights.device
        x1 = torch.tensor(rule1.feature_vector, dtype=torch.float32, device=device)
        x2 = torch.tensor(rule2.feature_vector, dtype=torch.float32, device=device)
        ci = (
            ci_value.to(device)
            if isinstance(ci_value, torch.Tensor)
            else torch.tensor(float(ci_value), dtype=torch.float32, device=device)
        )
        inv_diag = torch.reciprocal(self.sigma_diag.abs() + 1e-3)
        diff = x1 - x2
        kernel = torch.exp(-0.5 * torch.sum(diff * diff * inv_diag))
        authority = torch.tensor(
            min(rule1.authority_score, rule2.authority_score),
            dtype=torch.float32,
            device=device,
        )
        temporal = torch.tensor(
            CCSCalculator.compute_temporal_drift(rule1, rule2),
            dtype=torch.float32,
            device=device,
        )
        w_ci, w_k, w_h, w_td = torch.unbind(self.weights)
        z = w_ci * ci + w_k * kernel + w_h * authority - w_td * temporal
        return torch.sigmoid(z)

    def _build_knn_graph(
        self, corpus: Sequence[LegalRule], symbolic: SymbolicConflictChecker, k: int = 5
    ) -> torch.Tensor:
        if not corpus:
            return torch.zeros((0, 0), dtype=torch.float32)

        features = np.vstack([rule.feature_vector for rule in corpus])
        n = features.shape[0]
        adjacency = np.zeros((n, n), dtype=np.float32)
        for idx in range(n):
            distances = np.linalg.norm(features - features[idx], axis=1)
            neighbor_idx = np.argsort(distances)[1 : min(k + 1, n)]
            for j in neighbor_idx:
                ci = symbolic.check_conflict(corpus[idx], corpus[j])
                if float(ci) >= 1.0:
                    continue
                weight = float(np.exp(-0.5 * (distances[j] ** 2)))
                adjacency[idx, j] = max(adjacency[idx, j], weight)
                adjacency[j, idx] = max(adjacency[j, idx], weight)
        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency
        return torch.tensor(laplacian, dtype=torch.float32)

    @staticmethod
    def build_pairs_from_corpus(
        corpus: Sequence[LegalRule],
        labels: Iterable[Tuple[str, str, float]],
    ) -> List[Tuple[LegalRule, LegalRule, float]]:
        mapping = {rule.id: rule for rule in corpus}
        results: List[Tuple[LegalRule, LegalRule, float]] = []
        for rid1, rid2, value in labels:
            r1 = mapping.get(rid1)
            r2 = mapping.get(rid2)
            if r1 is None or r2 is None:
                continue
            results.append((r1, r2, float(value)))
        return results

    def train(
        self,
        dataset: LegalConflictDataset,
        *,
        symbolic: SymbolicConflictChecker,
        corpus: Optional[Sequence[LegalRule]] = None,
        num_epochs: int = 5,
        use_ssl: bool = False,
        use_graph: bool = False,
        unlabeled_pairs: Optional[Sequence[Tuple[LegalRule, LegalRule]]] = None,
        save_path: Path | str = Path("models/cale_weights.pt"),
    ) -> Dict[str, List[float]]:
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        history: Dict[str, List[float]] = {"supervised": [], "ssl": [], "graph": []}

        corpus = corpus or []
        laplacian = None
        if use_graph and corpus:
            laplacian = self._build_knn_graph(corpus, symbolic)

        unlabeled = list(unlabeled_pairs or [])
        if not unlabeled:
            unlabeled = [(r1, r2) for r1, r2, *_ in dataset]

        for epoch in range(num_epochs):
            epoch_sup: List[float] = []
            epoch_ssl: List[float] = []
            for rule1, rule2, label, ci in dataset:
                optimizer.zero_grad()
                target = torch.tensor(float(label), dtype=torch.float32, device=self.weights.device)
                ci_tensor = torch.tensor(float(ci), dtype=torch.float32, device=self.weights.device)
                pred = self.forward(rule1, rule2, ci_tensor)
                sup_loss = self.loss.compute_supervised_loss(pred, target)
                total_loss = sup_loss
                ssl_loss_val = torch.tensor(0.0, dtype=torch.float32, device=self.weights.device)
                if use_ssl and unlabeled:
                    pseudo_conf = pred.detach()
                    confidence = torch.clamp(torch.abs(pseudo_conf - 0.5) * 2.0, 0.0, 1.0)
                    ssl_loss_val = self.loss.compute_ssl_loss(pred, pseudo_conf, confidence)
                    total_loss = total_loss + 0.1 * ssl_loss_val
                l1_loss = self.loss.compute_l1_loss(self.weights, self.sigma_diag)
                total_loss = total_loss + 1e-3 * l1_loss
                total_loss.backward()
                clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_sup.append(float(sup_loss.detach().cpu()))
                if use_ssl:
                    epoch_ssl.append(float(ssl_loss_val.detach().cpu()))

            graph_loss_val = 0.0
            if use_graph and laplacian is not None and laplacian.numel() > 0:
                optimizer.zero_grad()
                if corpus:
                    anchor = corpus[0]
                    ccs_values: List[torch.Tensor] = []
                    for rule in corpus:
                        ci_anchor = symbolic.check_conflict(rule, anchor)
                        ci_tensor = torch.tensor(float(ci_anchor), dtype=torch.float32, device=self.weights.device)
                        score = self.forward(rule, anchor, ci_tensor)
                        ccs_values.append(score)
                    ccs_vec = torch.stack(ccs_values)
                    graph_loss = self.loss.compute_graph_loss(ccs_vec, laplacian.to(self.weights.device))
                    (0.05 * graph_loss).backward()
                    clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    graph_loss_val = float(graph_loss.detach().cpu())
            history["supervised"].append(float(np.mean(epoch_sup) if epoch_sup else 0.0))
            history["ssl"].append(float(np.mean(epoch_ssl) if epoch_ssl else 0.0))
            history["graph"].append(graph_loss_val)

        self._save_weights(save_path)
        return history

    def _save_weights(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights.detach().cpu(),
            "sigma_diag": self.sigma_diag.detach().cpu(),
        }
        torch.save(payload, destination)

    def export_weights(self) -> List[float]:
        return [float(value) for value in self.weights.detach().cpu().tolist()]


__all__ = [
    "CALELoss",
    "CALETrainer",
    "LegalConflictDataset",
]
