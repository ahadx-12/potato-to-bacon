"""Parser tests for CALE."""

from __future__ import annotations

import pytest

from potatobacon.cale.parser import PredicateMapper, RuleParser
from potatobacon.cale.types import ParseMetadata


def _metadata() -> ParseMetadata:
    return ParseMetadata(jurisdiction="Example", statute="Statute", section="1", enactment_year=2024)


def test_parser_must_vs_cannot_same_action_condition_extraction() -> None:
    mapper = PredicateMapper()
    parser = RuleParser(mapper)
    metadata = _metadata()

    rule1 = parser.parse("Organizations MUST collect personal data IF consent.", metadata)
    rule2 = parser.parse("Organizations CANNOT collect personal data UNLESS emergency.", metadata)

    assert rule1.modality == "OBLIGE"
    assert rule2.modality == "FORBID"
    assert rule1.action == rule2.action == "collect_personal_data"
    assert rule1.conditions == ["consent"]
    assert rule2.conditions == ["¬emergency"]


def test_predicate_mapper_is_deterministic() -> None:
    mapper = PredicateMapper()

    assert mapper.canonicalize_condition("IF Consent") == "consent"
    assert mapper.canonicalize_condition("if consent") == "consent"
    assert mapper.canonicalize_condition("CONSENT") == "consent"
    assert mapper.canonicalize_condition("UNLESS Emergency") == "¬emergency"


def test_parser_rejects_missing_modality() -> None:
    mapper = PredicateMapper()
    parser = RuleParser(mapper)

    with pytest.raises(ValueError):
        parser.parse("Organizations collect personal data if consent.", _metadata())
