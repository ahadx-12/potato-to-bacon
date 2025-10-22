"""Core constants for the Context-Aware Legal Engine (CALE).

The constants defined here cover modality normalization, condition keyword
handling, canonical stop words for action extraction, and other tiny helper
utilities that make the parser deterministic.  The module is intentionally
minimal so it can be imported without side effects.
"""

from __future__ import annotations

from typing import Mapping

# --- Modality handling ----------------------------------------------------

#: Mapping of textual modality markers to their canonical CALE modality token.
#: The keys are upper-cased for fast case-insensitive lookup.
MODALITY_NORMALIZATION: Mapping[str, str] = {
    "MUST": "OBLIGE",
    "SHALL": "OBLIGE",
    "MUST NOT": "FORBID",
    "SHALL NOT": "FORBID",
    "CANNOT": "FORBID",
    "CAN NOT": "FORBID",
    "MAY": "PERMIT",
    "IS PERMITTED TO": "PERMIT",
    "CAN": "PERMIT",
}

# --- Condition parsing ----------------------------------------------------

#: Ordered tuple of keywords that introduce rule conditions.  The order matters
#: when scanning text because multi-word markers must be evaluated before their
#: prefixes (e.g. ``EXCEPT WHEN`` before ``IF``).
CONDITION_KEYWORDS: tuple[str, ...] = (
    "EXCEPT WHEN",
    "PROVIDED THAT",
    "UNLESS",
    "IF",
)

# --- Action canonicalisation ----------------------------------------------

#: Stop words that should be dropped when normalising an action into a
#: snake_case verb phrase.  The list intentionally mirrors typical auxiliary
#: verbs and determiners found in legal drafting.
CANONICAL_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "may",
        "must",
        "not",
        "of",
        "or",
        "shall",
        "should",
        "such",
        "that",
        "the",
        "their",
        "to",
        "will",
    }
)

# --- Jurisdiction canonicalisation ----------------------------------------


def canonicalize_jurisdiction(name: str) -> str:
    """Canonicalise the provided jurisdiction string.

    The transformation is intentionally conservative; it strips leading/trailing
    whitespace, lower-cases the input, and then title-cases it.  The helper is a
    placeholder for richer logic in later CALE iterations but keeps the API
    stable for callers that already depend on canonical values.

    Parameters
    ----------
    name:
        Arbitrary jurisdiction name.

    Returns
    -------
    str
        A title-cased canonical representation.
    """

    return name.strip().lower().title()


# --- Predicate synonyms ----------------------------------------------------

#: A small set of canonical condition synonyms used by the predicate mapper.
#: Keys are arbitrary textual forms while values are canonical slugs.
CONDITION_SYNONYMS: Mapping[str, str] = {
    "consent": "consent",
    "emergency": "emergency",
    "national security": "national_security",
    "national-security": "national_security",
}

