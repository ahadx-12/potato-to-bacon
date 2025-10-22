"""Fundamental datatypes for CALE.

The data structures in this module are intentionally lightweight and highly
structured so that the parser, symbolic engine, and downstream components can
share a single canonical representation of legal rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict


@dataclass(slots=True)
class LegalRule:
    """Canonical representation of a parsed legal rule augmented with features.

    Parameters
    ----------
    id:
        Deterministic identifier for the rule.
    text:
        Raw rule text that was parsed.
    subject:
        The grammatical subject, lightly normalised.
    modality:
        Canonical modality token (``"OBLIGE"``, ``"FORBID"``, ``"PERMIT"``).
    action:
        Canonical snake_case action phrase.
    conditions:
        Sorted list of canonical condition literals.  Positive atoms are stored as
        ``"token"`` while negated forms use the logical negation symbol
        ``"¬token"``.
    jurisdiction, statute, section, enactment_year:
        Contextual metadata copied verbatim from the source document.
    interpretive_vec, situational_vec, temporal_scalar, jurisdictional_vec:
        Feature vectors populated by :class:`~potatobacon.cale.embed.FeatureEngine`.
        They default to ``None`` until the feature engine enriches the rule.
    authority_score:
        Normalised authority weight derived from the citation graph.
    """

    id: str
    text: str
    subject: str
    modality: str
    action: str
    conditions: List[str]
    jurisdiction: str
    statute: str
    section: str
    enactment_year: int
    interpretive_vec: np.ndarray | None = field(default=None, repr=False)
    situational_vec: np.ndarray | None = field(default=None, repr=False)
    temporal_scalar: float | None = None
    jurisdictional_vec: np.ndarray | None = field(default=None, repr=False)
    authority_score: float = 0.0

    @property
    def enactment_date(self) -> int:
        """Alias for ``enactment_year`` to match downstream expectations."""

        return self.enactment_year

    @enactment_date.setter
    def enactment_date(self, value: int) -> None:
        self.enactment_year = int(value)

    @property
    def feature_vector(self) -> np.ndarray:
        """Concatenate all feature components into a single vector.

        Raises
        ------
        ValueError
            If any feature component has not been populated yet.
        """

        if (
            self.situational_vec is None
            or self.interpretive_vec is None
            or self.jurisdictional_vec is None
            or self.temporal_scalar is None
        ):
            raise ValueError("LegalRule is missing feature components")
        parts = (
            np.asarray(self.situational_vec, dtype=np.float32),
            np.asarray(self.interpretive_vec, dtype=np.float32),
            np.asarray([self.temporal_scalar], dtype=np.float32),
            np.asarray(self.jurisdictional_vec, dtype=np.float32),
        )
        return np.concatenate(parts, dtype=np.float32)


@dataclass(slots=True)
class ParseMetadata:
    """Metadata required for parsing a legal rule text."""

    jurisdiction: str
    statute: str
    section: str
    enactment_year: int
    id: str | None = None


@dataclass(slots=True)
class ConflictAnalysis:
    """Container returned by the CCS calculator with helpful statistics."""

    rule1: LegalRule
    rule2: LegalRule
    CI: float
    K: float
    H: float
    TD: float
    CCS_textualist: float
    CCS_living: float
    CCS_pragmatic: float

    @property
    def scores(self) -> Sequence[float]:
        return (self.CCS_textualist, self.CCS_living, self.CCS_pragmatic)

    @property
    def variance(self) -> float:
        return float(np.var(self.scores))

    @property
    def interpretation(self) -> str:
        mean_score = float(np.mean(self.scores))
        if mean_score >= 0.75:
            return "High conflict risk"
        if mean_score >= 0.4:
            return "Moderate conflict risk"
        return "Low conflict risk"


class RuleInput(BaseModel):
    """Pydantic schema for externally provided rule data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str
    jurisdiction: str
    statute: str
    section: str
    enactment_year: int

