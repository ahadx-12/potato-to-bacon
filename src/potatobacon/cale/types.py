"""Fundamental datatypes for CALE.

The data structures in this module are intentionally lightweight and highly
structured so that the parser, symbolic engine, and downstream components can
share a single canonical representation of legal rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, ConfigDict


@dataclass(slots=True)
class LegalRule:
    """Canonical representation of a parsed legal rule.

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


@dataclass(slots=True)
class ParseMetadata:
    """Metadata required for parsing a legal rule text."""

    jurisdiction: str
    statute: str
    section: str
    enactment_year: int


class RuleInput(BaseModel):
    """Pydantic schema for externally provided rule data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str
    jurisdiction: str
    statute: str
    section: str
    enactment_year: int

