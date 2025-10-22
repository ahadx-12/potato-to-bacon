"""Gradient-guided amendment suggester for CALE."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - maintain deterministic fallback
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - prefer scikit-learn implementation
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover - lightweight fallback
    class NearestNeighbors:  # type: ignore[too-many-ancestors]
        def __init__(self, n_neighbors: int, metric: str = "euclidean") -> None:
            self.n_neighbors = n_neighbors
            self.metric = metric
            self._data: np.ndarray | None = None

        def fit(self, X: np.ndarray) -> "NearestNeighbors":
            self._data = np.asarray(X, dtype=np.float32)
            return self

        def kneighbors(
            self, X: np.ndarray, n_neighbors: int | None = None
        ) -> Tuple[np.ndarray, np.ndarray]:
            if self._data is None:
                raise RuntimeError("NearestNeighbors not fitted")
            query = np.asarray(X, dtype=np.float32)
            n = n_neighbors or self.n_neighbors
            dists = []
            indices = []
            for row in query:
                diff = self._data - row
                dist = np.linalg.norm(diff, axis=1)
                order = np.argsort(dist)[:n]
                dists.append(dist[order])
                indices.append(order)
            return np.asarray(dists), np.asarray(indices)

from .ccs import CCSCalculator
from .embed import LegalEmbedder
from .parser import PredicateMapper
from .types import ConflictAnalysis, LegalRule

SEED = int(os.getenv("CALE_SEED", "1337"))
random.seed(SEED)
np.random.seed(SEED)
if torch is not None:  # pragma: no branch
    try:  # pragma: no cover - guard against broken torch installs
        torch.manual_seed(SEED)
    except Exception:
        pass


@dataclass
class Suggestion:
    """Structured amendment suggestion with lightweight metadata."""

    condition: str
    justification: Dict[str, float]
    estimated_ccs: float
    suggested_text: str


class AmendmentSuggester:
    """Gradient-guided amendment proposer with kNN precedent retrieval.

    Deterministic and CPU-only. Works without GPUs.
    """

    def __init__(
        self,
        rule_corpus: List[LegalRule],
        embedder: LegalEmbedder,
        ccs_calc: CCSCalculator,
        predicate_mapper: PredicateMapper,
        k_neighbors: int = 8,
    ) -> None:
        self.embedder = embedder
        self.ccs = ccs_calc
        self.mapper = predicate_mapper
        self.corpus = rule_corpus
        self.k = int(max(1, k_neighbors))
        self._fit_knn()

    def _fit_knn(self) -> None:
        if not self.corpus:
            self.knn: NearestNeighbors | None = None
            return
        vectors = [rule.feature_vector for rule in self.corpus]
        X = np.vstack(vectors)
        self.knn = NearestNeighbors(
            n_neighbors=min(self.k * 3, len(self.corpus)), metric="euclidean"
        )
        self.knn.fit(X)

    # ---------- Gradient ----------
    def _ccs_grad_x1(self, r1: LegalRule, r2: LegalRule, CI: float) -> np.ndarray:
        """Return ∂CCS/∂x1 using torch autograd on the pragmatic Σ."""

        if torch is None:
            x1 = r1.feature_vector
            x2 = r2.feature_vector
            eps = 1e-4
            base = self.ccs.compute_ccs(r1, r2, CI, philosophy="pragmatic")
            grad = np.zeros_like(x1)
            for idx in range(x1.size):
                x1_eps = x1.copy()
                x1_eps[idx] += eps
                perturbed = self._with_vector(r1, x1_eps)
                grad[idx] = (
                    self.ccs.compute_ccs(perturbed, r2, CI, "pragmatic") - base
                ) / eps
            return grad

        x1 = torch.tensor(r1.feature_vector, dtype=torch.float32, requires_grad=True)
        x2 = torch.tensor(r2.feature_vector, dtype=torch.float32)
        z, _ = self.ccs._torch_ccs_components(x1, x2, CI)
        ccs = torch.sigmoid(z)
        ccs.backward()
        grad = x1.grad.detach().cpu().numpy()
        return grad

    def _with_vector(self, rule: LegalRule, vector: np.ndarray) -> LegalRule:
        """Return a copy of ``rule`` with ``feature_vector`` replaced by ``vector``."""

        import copy

        clone = copy.deepcopy(rule)
        ds = len(clone.situational_vec)
        di = len(clone.interpretive_vec)
        dt = 1
        clone.situational_vec = vector[:ds]
        clone.interpretive_vec = vector[ds : ds + di]
        clone.temporal_scalar = float(vector[ds + di])
        clone.jurisdictional_vec = vector[ds + di + dt :]
        return clone

    def _low_conflict_target(
        self, r1: LegalRule, r2: LegalRule, CI: float, alpha: float = 0.5
    ) -> np.ndarray:
        gradient = self._ccs_grad_x1(r1, r2, CI)
        norm = float(np.linalg.norm(gradient)) + 1e-12
        alpha = float(np.clip(alpha, 0.2, 0.8))
        return r1.feature_vector - alpha * (gradient / norm)

    # ---------- kNN retrieval ----------
    def _find_precedents(
        self,
        x_target: np.ndarray,
        r1: LegalRule,
        r2: LegalRule,
        current_ccs: float,
        CI: float,
        topk: int = 5,
    ) -> List[LegalRule]:
        if self.knn is None or not self.corpus:
            return []
        neighbors = min(self.k * 3, len(self.corpus))
        dists, idx = self.knn.kneighbors([x_target], n_neighbors=neighbors)
        results: List[LegalRule] = []
        anchor = r1.interpretive_vec
        norm_anchor = float(np.linalg.norm(anchor)) + 1e-12
        for pos in idx[0]:
            candidate = self.corpus[int(pos)]
            if candidate.id == r1.id:
                continue
            if set(candidate.conditions) == set(r1.conditions):
                continue
            i1 = candidate.interpretive_vec
            cos = float(np.dot(i1, anchor) / ((np.linalg.norm(i1) * norm_anchor) + 1e-12))
            if cos < 0.70:
                continue
            ccs_candidate = self.ccs.compute_ccs(candidate, r2, CI, philosophy="pragmatic")
            if ccs_candidate < current_ccs:
                results.append(candidate)
            if len(results) >= topk:
                break
        return results

    # ---------- Condition mining ----------
    def _extract_candidates(
        self, r1: LegalRule, precedents: List[LegalRule]
    ) -> List[str]:
        base = {self.mapper.normalize_condition(cond) for cond in r1.conditions}
        bag: List[str] = []
        for precedent in precedents:
            for cond in precedent.conditions:
                normalized = self.mapper.normalize_condition(cond)
                if normalized not in base:
                    bag.append(normalized)
        return bag

    def _embed_condition(self, condition: str) -> np.ndarray:
        return self.embedder.embed_phrase(condition)

    def _estimate_reduction(
        self, r1: LegalRule, r2: LegalRule, condition: str, CI: float
    ) -> float:
        import copy

        augmented = copy.deepcopy(r1)
        if condition.startswith("¬"):
            new_condition = condition
        else:
            new_condition = f"¬{condition}"
        augmented.conditions = list(r1.conditions) + [new_condition]
        augmented.text = f"{r1.text} EXCEPT WHEN {condition}".strip()
        augmented.situational_vec = self.embedder.compute_situational_vec(augmented)
        augmented.interpretive_vec = self.embedder.embed_rule(augmented)
        new_ccs = self.ccs.compute_ccs(augmented, r2, CI, philosophy="pragmatic")
        baseline = self.ccs.compute_ccs(r1, r2, CI, philosophy="pragmatic")
        return max(0.0, baseline - new_ccs)

    def _rank(
        self, r1: LegalRule, r2: LegalRule, CI: float, candidates: List[str]
    ) -> List[Suggestion]:
        if not candidates:
            return []
        freq: Dict[str, int] = {}
        for cond in candidates:
            freq[cond] = freq.get(cond, 0) + 1
        total = len(candidates)
        suggestions: List[Suggestion] = []
        interpretive = r1.interpretive_vec
        norm_interpretive = float(np.linalg.norm(interpretive)) + 1e-12
        baseline_ccs = self.ccs.compute_ccs(r1, r2, CI, philosophy="pragmatic")
        for cond, count in freq.items():
            embedding = self._embed_condition(cond)
            relevance = float(
                np.dot(embedding, interpretive)
                / ((np.linalg.norm(embedding) * norm_interpretive) + 1e-12)
            )
            impact = float(self._estimate_reduction(r1, r2, cond, CI))
            score = 0.20 * (count / total) + 0.30 * relevance + 0.50 * impact
            suggestions.append(
                Suggestion(
                    condition=cond,
                    justification={
                        "frequency": count / total,
                        "semantic_relevance": relevance,
                        "impact": impact,
                        "composite_score": score,
                    },
                    estimated_ccs=float(baseline_ccs - impact),
                    suggested_text=f"{r1.text} EXCEPT WHEN {cond}",
                )
            )
        suggestions.sort(key=lambda item: item.justification["composite_score"], reverse=True)
        return suggestions

    # ---------- Public API ----------
    def suggest_amendment(
        self, r1: LegalRule, r2: LegalRule, analysis: ConflictAnalysis, k: int = 5
    ) -> Dict[str, object]:
        CI = float(analysis.CI)
        current_ccs = float(analysis.CCS_pragmatic)
        target = self._low_conflict_target(r1, r2, CI)
        precedents = self._find_precedents(target, r1, r2, current_ccs, CI, topk=max(1, k))
        candidates = self._extract_candidates(r1, precedents)
        ranked = self._rank(r1, r2, CI, candidates)
        top_three = ranked[:3]
        best = top_three[0] if top_three else None
        return {
            "precedent_count": len(precedents),
            "candidates_considered": len(candidates),
            "suggestions": [s.__dict__ for s in top_three],
            "best": (best.__dict__ if best else None),
        }
