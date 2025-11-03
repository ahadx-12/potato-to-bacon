"""Lightweight precedent index backed by FAISS (if available) or NumPy fallbacks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - gracefully degrade to numpy backend
    faiss = None  # type: ignore[assignment]

from .types import LegalRule
from .embed import LegalEmbedder


@dataclass(slots=True)
class PrecedentCase:
    """Representation of a curated precedent."""

    id: str
    title: str
    citation: str
    summary: str
    excerpt: str
    vector: np.ndarray


def _normalise(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr)) or 1.0
    return arr / norm


def _case_vector(payload: dict, embedder: LegalEmbedder) -> np.ndarray:
    vector = payload.get("vector")
    if vector is not None:
        return _normalise(np.asarray(vector, dtype=np.float32))
    basis = payload.get("summary") or payload.get("excerpt") or payload.get("title")
    if not basis:
        raise ValueError(f"Precedent entry {payload.get('id')} missing text basis")
    return _normalise(embedder.embed_phrase(basis))


def load_precedent_cases(
    source: str | Path,
    *,
    embedder: LegalEmbedder,
) -> List[PrecedentCase]:
    """Load precedent metadata and vectors from ``source``."""

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Precedent corpus missing at {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw: Sequence[dict] = json.load(handle)

    cases: List[PrecedentCase] = []
    for payload in raw:
        vector = _case_vector(payload, embedder)
        cases.append(
            PrecedentCase(
                id=str(payload.get("id")),
                title=str(payload.get("title", "")),
                citation=str(payload.get("citation", "")),
                summary=str(payload.get("summary", "")),
                excerpt=str(payload.get("excerpt", "")),
                vector=vector,
            )
        )
    return cases


class PrecedentIndex:
    """Similarity search over curated disputes."""

    def __init__(self, cases: Iterable[PrecedentCase]):
        self._cases: List[PrecedentCase] = list(cases)
        if not self._cases:
            self._vectors = np.zeros((0, 1), dtype=np.float32)
            self._index = None
            self._use_faiss = False
            return
        self._vectors = np.vstack([case.vector for case in self._cases]).astype(np.float32)
        self._use_faiss = bool(faiss)
        if self._use_faiss:
            dim = int(self._vectors.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(self._vectors)
            self._index = index
        else:
            self._index = None

    def is_empty(self) -> bool:
        return not self._cases

    def _query_vector(self, rule1: LegalRule, rule2: LegalRule) -> np.ndarray:
        vec1 = getattr(rule1, "interpretive_vec", None)
        vec2 = getattr(rule2, "interpretive_vec", None)
        if vec1 is None or vec2 is None:
            return np.zeros(self._vectors.shape[1], dtype=np.float32)
        combined = (np.asarray(vec1, dtype=np.float32) + np.asarray(vec2, dtype=np.float32)) / 2.0
        return _normalise(combined)

    def search(
        self, rule1: LegalRule, rule2: LegalRule, *, top_k: int = 5
    ) -> List[dict]:
        """Return the ``top_k`` closest precedents for ``(rule1, rule2)``."""

        if self.is_empty():
            return []
        query = self._query_vector(rule1, rule2)
        if not query.any():
            return []
        if self._use_faiss and self._index is not None:
            scores, idx = self._index.search(query.reshape(1, -1), top_k)
            similarities = scores[0]
            indices = idx[0]
        else:
            similarities = self._vectors @ query
            indices = np.argsort(similarities)[::-1][:top_k]
        results: List[dict] = []
        for rank, case_idx in enumerate(indices):
            if case_idx < 0 or case_idx >= len(self._cases):
                continue
            case = self._cases[int(case_idx)]
            if self._use_faiss:
                similarity = float(similarities[rank] if rank < len(similarities) else 0.0)
            else:
                similarity = float(similarities[int(case_idx)])
            results.append(
                {
                    "id": case.id,
                    "title": case.title,
                    "citation": case.citation,
                    "excerpt": case.excerpt,
                    "summary": case.summary,
                    "similarity": float(max(-1.0, min(1.0, similarity))),
                }
            )
        return results


__all__ = ["PrecedentCase", "PrecedentIndex", "load_precedent_cases"]
