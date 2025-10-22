"""Context-Aware Legal Engine (CALE).

Days 1–3 expose the foundational building blocks: constants, datatypes,
parsing utilities, and the symbolic conflict checker.
"""

from __future__ import annotations

from .constants import (
    CANONICAL_STOPWORDS,
    CONDITION_KEYWORDS,
    CONDITION_SYNONYMS,
    MODALITY_NORMALIZATION,
    canonicalize_jurisdiction,
)
from .embed import FeatureEngine, LegalEmbedder
from .graph import compute_authority_scores, load_citation_graph
from .parser import PredicateMapper, RuleParser
from .suggest import AmendmentSuggester, Suggestion
from .symbolic import SymbolicConflictChecker
from .types import ConflictAnalysis, LegalRule, ParseMetadata, RuleInput

__all__ = [
    "CANONICAL_STOPWORDS",
    "CONDITION_KEYWORDS",
    "CONDITION_SYNONYMS",
    "MODALITY_NORMALIZATION",
    "FeatureEngine",
    "LegalEmbedder",
    "compute_authority_scores",
    "load_citation_graph",
    "PredicateMapper",
    "RuleParser",
    "AmendmentSuggester",
    "Suggestion",
    "SymbolicConflictChecker",
    "ConflictAnalysis",
    "LegalRule",
    "ParseMetadata",
    "RuleInput",
    "canonicalize_jurisdiction",
]

