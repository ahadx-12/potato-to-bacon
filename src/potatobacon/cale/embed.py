"""Deterministic embedding and feature generation for CALE."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import replace
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover - torch optional in CI
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - maintain deterministic fallback
    torch = None  # type: ignore[assignment]

SEED = int(os.getenv("CALE_SEED", "1337"))
random.seed(SEED)
np.random.seed(SEED)
if torch is not None:  # pragma: no branch
    try:  # pragma: no cover - guard against broken torch installs
        torch.manual_seed(SEED)
    except Exception:
        pass

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import guard
    SentenceTransformer = None  # type: ignore[assignment]

from .types import LegalRule

# Fixed registries keep one-hot sizes stable across runs
JURISDICTIONS = [
    "CA.Federal",
    "US.Federal",
    "EU",
    "CA.BC",
    "CA.ON",
    "UK",
]
JURISDICTION_ALIASES = {
    "Canada.Federal": "CA.Federal",
    "Canada.BC": "CA.BC",
    "Canada.ON": "CA.ON",
    "United States.Federal": "US.Federal",
    "U.S.Federal": "US.Federal",
}
JURISDICTION_INDEX = {j: i for i, j in enumerate(JURISDICTIONS)}

D_S = 4  # [requires_consent, emergency_exception, commercial_context, data_sensitivity]


class _HashBackend:
    """Deterministic "embedding" by hashing text into a unit vector."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        tokens = [tok for tok in text.lower().split() if tok]
        if not tokens:
            base = hashlib.sha256(b"").digest()
            raw = np.frombuffer((base * ((self.dim // len(base)) + 1))[: self.dim], dtype=np.uint8)
            vec = (raw.astype(np.float32) - 127.5) / 127.5
            norm = np.linalg.norm(vec) or 1.0
            return vec / norm

        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            raw = (np.frombuffer(digest, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5
            for idx, value in enumerate(raw):
                vec[(idx) % self.dim] += value
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm


class _STBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)


class LegalEmbedder:
    """Produce interpretive embeddings for :class:`LegalRule` objects."""

    def __init__(self, *, backend: Optional[str] = None, dim_interpretive: int = 384, seed: int = 13):
        self.dim_interpretive = int(dim_interpretive)
        backend = backend or os.getenv("CALE_EMBED_BACKEND", "hash")
        self._rng = np.random.RandomState(seed)
        if backend == "st":
            self._backend = _STBackend()
        else:
            self._backend = _HashBackend(dim=self.dim_interpretive)

    def embed_rule(self, rule: LegalRule) -> np.ndarray:
        full_text = f"{rule.subject} {rule.modality} {rule.action}. {rule.text}".strip()
        return self._backend.encode(full_text)

    def embed_phrase(self, phrase: str) -> np.ndarray:
        """Embed an arbitrary phrase using the deterministic backend."""

        return self._backend.encode(phrase)

    @staticmethod
    def compute_situational_vec(rule: LegalRule) -> np.ndarray:
        text = (rule.text or "").lower()
        primary_text = text
        for marker in ("except when", "unless"):
            if marker in primary_text:
                primary_text = primary_text.split(marker)[0]
                break
        conds = [c.lower() for c in (rule.conditions or [])]
        def _has_positive(keyword: str) -> bool:
            return any(keyword in c and not c.startswith("¬") for c in conds)

        requires_consent = float(_has_positive("consent") or "consent" in primary_text)
        emergency_exception = float(_has_positive("emergency") or "emergency" in primary_text)
        commercial_context = float("commercial" in primary_text or "commerce" in primary_text)
        sensitive_terms = ("health", "financial", "biometric", "genetic", "political")
        data_sensitivity = float(sum(term in primary_text for term in sensitive_terms)) / len(sensitive_terms)
        vec = np.array(
            [requires_consent, emergency_exception, commercial_context, data_sensitivity],
            dtype=np.float32,
        )
        return vec

    @staticmethod
    def compute_temporal_scalar(year: int) -> float:
        year_int = max(1900, min(2100, int(year or 1900)))
        return float((year_int - 1900) / 200.0)

    @staticmethod
    def compute_jurisdictional_vec(jurisdiction: str) -> np.ndarray:
        canonical = JURISDICTION_ALIASES.get(jurisdiction, jurisdiction)
        vec = np.zeros(len(JURISDICTIONS), dtype=np.float32)
        idx = JURISDICTION_INDEX.get(canonical)
        if idx is not None:
            vec[idx] = 1.0
        return vec


class FeatureEngine:
    """Populate a :class:`LegalRule` with embeddings, features, and authority score."""

    def __init__(self, embedder: Optional[LegalEmbedder] = None) -> None:
        self.embedder = embedder or LegalEmbedder()

    def populate(self, rule: LegalRule, *, authorities: Optional[Dict[str, float]] = None) -> LegalRule:
        interpretive_vec = self.embedder.embed_rule(rule)
        situational_vec = LegalEmbedder.compute_situational_vec(rule)
        temporal_scalar = LegalEmbedder.compute_temporal_scalar(getattr(rule, "enactment_year", 1900))
        jurisdictional_vec = LegalEmbedder.compute_jurisdictional_vec(getattr(rule, "jurisdiction", ""))
        authority = 0.5
        if authorities is not None:
            authority = float(authorities.get(rule.id, 0.0))
        return replace(
            rule,
            interpretive_vec=np.asarray(interpretive_vec, dtype=np.float32),
            situational_vec=np.asarray(situational_vec, dtype=np.float32),
            temporal_scalar=float(temporal_scalar),
            jurisdictional_vec=np.asarray(jurisdictional_vec, dtype=np.float32),
            authority_score=float(authority),
        )

    def populate_features(
        self, rule: LegalRule, *, authorities: Optional[Dict[str, float]] = None
    ) -> LegalRule:
        """Alias for :meth:`populate` to match API naming conventions."""

        return self.populate(rule, authorities=authorities)

__all__ = [
    "FeatureEngine",
    "LegalEmbedder",
    "JURISDICTIONS",
    "JURISDICTION_ALIASES",
    "JURISDICTION_INDEX",
    "D_S",
]
