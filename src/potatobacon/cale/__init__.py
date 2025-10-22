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
from .parser import PredicateMapper, RuleParser
from .symbolic import SymbolicConflictChecker
from .types import LegalRule, ParseMetadata, RuleInput

__all__ = [
    "CANONICAL_STOPWORDS",
    "CONDITION_KEYWORDS",
    "CONDITION_SYNONYMS",
    "MODALITY_NORMALIZATION",
    "PredicateMapper",
    "RuleParser",
    "SymbolicConflictChecker",
    "LegalRule",
    "ParseMetadata",
    "RuleInput",
    "canonicalize_jurisdiction",
]

