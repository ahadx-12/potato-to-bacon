"""Deterministic parser for the Context-Aware Legal Engine (CALE)."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import re

try:  # pragma: no cover - optional dependency
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


try:  # pragma: no cover - spaCy is optional
    import spacy
except Exception:  # pragma: no cover - spaCy absent or misconfigured
    spacy = None  # type: ignore[assignment]

from .constants import (
    CANONICAL_STOPWORDS,
    CONDITION_KEYWORDS,
    CONDITION_SYNONYMS,
    MODALITY_NORMALIZATION,
)
from .types import LegalRule, ParseMetadata, RuleInput


_NEGATION_MARKERS = ("UNLESS", "EXCEPT WHEN")
_POSITIVE_MARKERS = ("IF", "PROVIDED THAT")
_SUBJECT_DETERMINERS = {"the", "a", "an", "any", "all", "every", "each"}


@dataclass(slots=True)
class _ConditionFragment:
    marker: str
    text: str


class PredicateMapper:
    """Map raw condition text into canonical Boolean predicate names.

    The mapper is deterministic: any given raw fragment always maps to the same
    canonical atom.  Synonym mappings are applied before the slug is returned and
    the mapper keeps track of every atom it has produced via :pyattr:`seen_atoms`.
    """

    def __init__(
        self,
        synonyms: Mapping[str, str] | None = None,
    ) -> None:
        merged_synonyms: Dict[str, str] = dict(CONDITION_SYNONYMS)
        if synonyms:
            merged_synonyms.update(synonyms)

        self._synonyms: Dict[str, str] = {}
        for key, value in merged_synonyms.items():
            normalised_key = self._normalise_key(key)
            self._synonyms[normalised_key] = self._slug(value)
        self._atom_cache: MutableMapping[str, str] = {}
        self._seen_atoms: Dict[str, None] = {}

    @staticmethod
    def _normalise_key(raw: str) -> str:
        slug = re.sub(r"[^\w]+", " ", raw.lower()).strip()
        return slug

    @staticmethod
    def _strip_marker(raw: str) -> tuple[str, bool]:
        text = raw.strip()
        upper = text.upper()
        for marker in _NEGATION_MARKERS:
            if upper.startswith(marker):
                remainder = text[len(marker) :].strip(" ,.:;")
                return remainder, True
        for marker in _POSITIVE_MARKERS:
            if upper.startswith(marker):
                remainder = text[len(marker) :].strip(" ,.:;")
                return remainder, False
        return text.strip(" ,.:;"), False

    @staticmethod
    def _slug(raw: str) -> str:
        cleaned = re.sub(r"[^\w]+", " ", raw.lower()).strip()
        if not cleaned:
            raise ValueError("Condition phrase must contain alphanumeric characters")
        return re.sub(r"\s+", "_", cleaned)

    def canonical_atom(self, raw_positive_phrase: str) -> str:
        """Convert an arbitrary positive condition phrase into a canonical atom."""

        key = self._normalise_key(raw_positive_phrase)
        if key in self._atom_cache:
            return self._atom_cache[key]

        if key in self._synonyms:
            atom = self._synonyms[key]
        else:
            atom = self._slug(key)
        self._atom_cache[key] = atom
        self._seen_atoms.setdefault(atom, None)
        return atom

    def canonicalize_condition(self, raw: str) -> str:
        """Canonicalise a condition fragment into ``token`` or ``¬token``.

        Parameters
        ----------
        raw:
            Raw textual clause, potentially including markers such as ``IF`` or
            ``UNLESS``.  The method tolerates leading punctuation and whitespace
            and is idempotent.
        """

        fragment, negated = self._strip_marker(raw)
        atom = self.canonical_atom(fragment)
        literal = atom if not negated else f"¬{atom}"
        return literal

    @property
    def seen_atoms(self) -> Sequence[str]:
        """Return the atoms seen so far in lexical order."""

        return tuple(sorted(self._seen_atoms))

    def normalize_condition(self, raw: str) -> str:
        """Idempotently normalise an existing condition literal or fragment."""

        literal = raw.strip()
        if literal.startswith("¬"):
            atom = self.canonical_atom(literal[1:])
            return f"¬{atom}"
        return self.canonicalize_condition(literal)


class RuleParser:
    """Parse natural language rule sentences into :class:`LegalRule` objects."""

    def __init__(self, predicate_mapper: PredicateMapper) -> None:
        self._mapper = predicate_mapper
        self._nlp = None
        if spacy is not None:  # pragma: no cover - exercised only when spaCy exists
            try:
                self._nlp = spacy.blank("en")
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")
            except Exception:
                self._nlp = None

        keyword_pattern = "|".join(re.escape(keyword) for keyword in CONDITION_KEYWORDS)
        self._condition_regex = re.compile(rf"\b({keyword_pattern})\b", re.IGNORECASE)

        modalities = sorted(MODALITY_NORMALIZATION, key=len, reverse=True)
        pattern = "|".join(re.escape(token) for token in modalities)
        self._modality_regex = re.compile(rf"\b({pattern})\b", re.IGNORECASE)

    def parse(self, text: str, metadata: ParseMetadata | Mapping[str, Any]) -> LegalRule:
        """Parse *text* and attach :class:`ParseMetadata`.

        Raises
        ------
        ValueError
            If the parser cannot locate a modality or action within the text.
        """

        metadata_obj = self._coerce_metadata(metadata)
        clean_text = self._normalise_whitespace(text)
        modality_match = self._modality_regex.search(clean_text)
        if not modality_match:
            raise ValueError("No recognised modality token found in rule text")

        modality_token = modality_match.group(0)
        modality = MODALITY_NORMALIZATION[modality_token.upper()]

        subject_text = clean_text[: modality_match.start()].strip(" ,.")
        action_and_conditions = clean_text[modality_match.end() :].strip()

        subject = self._normalise_subject(subject_text)

        action_text, condition_text = self._split_action_and_conditions(action_and_conditions)
        action = self._canonical_action(action_text)
        conditions = self._parse_conditions(condition_text)

        rule_id = metadata_obj.id or self._rule_identifier(clean_text, metadata_obj)
        return LegalRule(
            id=rule_id,
            text=clean_text,
            subject=subject,
            modality=modality,
            action=action,
            conditions=conditions,
            jurisdiction=metadata_obj.jurisdiction,
            statute=metadata_obj.statute,
            section=metadata_obj.section,
            enactment_year=metadata_obj.enactment_year,
        )

    def from_input(self, rule_input: RuleInput) -> LegalRule:
        """Create a :class:`LegalRule` directly from a :class:`RuleInput`."""

        metadata = ParseMetadata(
            jurisdiction=rule_input.jurisdiction,
            statute=rule_input.statute,
            section=rule_input.section,
            enactment_year=rule_input.enactment_year,
        )
        return self.parse(rule_input.text, metadata)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def _coerce_metadata(metadata: ParseMetadata | Mapping[str, Any]) -> ParseMetadata:
        if isinstance(metadata, ParseMetadata):
            return metadata

        def _get(key: str, *aliases: str, default: Any | None = None) -> Any:
            keys = (key, *aliases)
            for candidate in keys:
                if candidate in metadata:
                    return metadata[candidate]
            if default is not None:
                return default
            raise KeyError(f"Missing metadata field: {key}")

        enactment_year = int(
            _get("enactment_year", "enactment_date", "year", default=1900)
        )
        raw_id = _get("id", default=None)
        identifier = None if raw_id in (None, "") else str(raw_id)
        return ParseMetadata(
            jurisdiction=str(_get("jurisdiction")),
            statute=str(_get("statute")),
            section=str(_get("section")),
            enactment_year=enactment_year,
            id=identifier,
        )

    @staticmethod
    def _rule_identifier(text: str, metadata: ParseMetadata) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"{metadata.statute}:{metadata.section}:{digest}"

    @staticmethod
    def _normalise_subject(subject_text: str) -> str:
        if not subject_text:
            return ""
        tokens = [token for token in re.split(r"\s+", subject_text) if token]
        while tokens and tokens[0].lower() in _SUBJECT_DETERMINERS:
            tokens.pop(0)
        subject = " ".join(tokens)
        return subject

    def _split_action_and_conditions(self, fragment: str) -> tuple[str, str]:
        if not fragment:
            raise ValueError("Rule text missing predicate after modality")
        match = self._condition_regex.search(fragment)
        if not match:
            return fragment, ""
        return fragment[: match.start()].strip(" ,"), fragment[match.start() :].strip()

    @staticmethod
    def _tokenise_action(action_text: str) -> List[str]:
        cleaned = re.sub(r"[^\w]+", " ", action_text.lower())
        tokens = [token for token in cleaned.split() if token]
        filtered = [token for token in tokens if token not in CANONICAL_STOPWORDS]
        return filtered or tokens

    def _canonical_action(self, action_text: str) -> str:
        if not action_text:
            raise ValueError("Unable to locate action phrase in rule text")
        tokens = self._tokenise_action(action_text)
        if not tokens:
            raise ValueError("Unable to canonicalise action phrase")
        return "_".join(tokens)

    def _parse_conditions(self, condition_text: str) -> List[str]:
        if not condition_text:
            return []
        fragments = self._extract_condition_fragments(condition_text)
        literals = [self._mapper.canonicalize_condition(f"{frag.marker} {frag.text}") for frag in fragments]
        unique = sorted(set(literals), key=lambda literal: (literal.lstrip("¬"), literal.startswith("¬")))
        return unique

    def _extract_condition_fragments(self, condition_text: str) -> List[_ConditionFragment]:
        matches = list(self._condition_regex.finditer(condition_text))
        if not matches:
            return []
        fragments: List[_ConditionFragment] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(condition_text)
            clause = condition_text[start:end].strip(" ,;.")
            if not clause:
                continue
            pieces = [piece.strip(" ,;.") for piece in re.split(r"(?i)\bAND\b|,|;", clause) if piece.strip()]
            if not pieces:
                pieces = [clause]
            for piece in pieces:
                fragments.append(_ConditionFragment(marker=match.group(0), text=piece))
        return fragments


__all__ = ["PredicateMapper", "RuleParser"]

