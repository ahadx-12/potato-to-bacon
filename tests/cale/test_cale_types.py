"""Tests for CALE core datatypes."""

from __future__ import annotations

from potatobacon.cale.types import LegalRule, ParseMetadata


def test_legalrule_dataclass_core_fields() -> None:
    rule = LegalRule(
        id="statute:sec:abc123",
        text="Org MUST collect personal data.",
        subject="Org",
        modality="OBLIGE",
        action="collect_personal_data",
        conditions=["consent"],
        jurisdiction="Example",
        statute="Statute",
        section="1",
        enactment_year=2024,
    )

    assert rule.modality in {"OBLIGE", "FORBID", "PERMIT"}
    assert isinstance(rule.conditions, list)
    assert rule.conditions == ["consent"]


def test_conditions_are_sorted_and_unique() -> None:
    metadata = ParseMetadata(
        jurisdiction="Example",
        statute="Statute",
        section="1",
        enactment_year=2024,
    )

    from potatobacon.cale.parser import PredicateMapper, RuleParser

    mapper = PredicateMapper()
    parser = RuleParser(mapper)
    rule = parser.parse(
        "Organizations MUST collect personal data IF consent AND IF consent, UNLESS emergency.",
        metadata,
    )

    assert rule.conditions == ["consent", "¬emergency"]
