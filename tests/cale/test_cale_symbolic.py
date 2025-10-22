"""Symbolic conflict checker tests."""

from __future__ import annotations

from typing import Tuple

import hypothesis.strategies as st
from hypothesis import given

from potatobacon.cale.parser import PredicateMapper, RuleParser
from potatobacon.cale.symbolic import SymbolicConflictChecker
from potatobacon.cale.types import ParseMetadata


def _parser_and_checker() -> Tuple[RuleParser, SymbolicConflictChecker, ParseMetadata]:
    mapper = PredicateMapper()
    parser = RuleParser(mapper)
    checker = SymbolicConflictChecker(mapper)
    metadata = ParseMetadata(jurisdiction="Example", statute="Statute", section="1", enactment_year=2024)
    return parser, checker, metadata


def _parse(parser: RuleParser, metadata: ParseMetadata, text: str):
    return parser.parse(text, metadata)


def test_direct_oblige_forbid_conflict() -> None:
    parser, checker, metadata = _parser_and_checker()
    r1 = _parse(parser, metadata, "Org MUST collect personal data IF consent.")
    r2 = _parse(parser, metadata, "Org CANNOT collect personal data IF consent.")

    assert checker.check_conflict(r1, r2) == 1.0


def test_permit_vs_forbid_conflict() -> None:
    parser, checker, metadata = _parser_and_checker()
    r1 = _parse(parser, metadata, "Org MAY collect personal data IF consent.")
    r2 = _parse(parser, metadata, "Org CANNOT collect personal data IF consent.")

    assert checker.check_conflict(r1, r2) == 1.0


def test_compatible_if_vs_unless_when_nonoverlapping() -> None:
    parser, checker, metadata = _parser_and_checker()
    r1 = _parse(parser, metadata, "Org MUST collect personal data IF consent.")
    r2 = _parse(parser, metadata, "Org CANNOT collect personal data UNLESS consent.")

    assert checker.check_conflict(r1, r2) == 0.0


def test_different_actions_do_not_conflict() -> None:
    parser, checker, metadata = _parser_and_checker()
    r1 = _parse(parser, metadata, "Org MUST encrypt records IF health.")
    r2 = _parse(parser, metadata, "Org CANNOT disclose records IF health.")

    assert checker.check_conflict(r1, r2) == 0.0


def test_identical_rules_do_not_conflict() -> None:
    parser, checker, metadata = _parser_and_checker()
    text = "Org MUST collect personal data IF consent."
    r1 = _parse(parser, metadata, text)
    r2 = _parse(parser, metadata, text)

    assert checker.check_conflict(r1, r2) == 0.0


@given(st.text(alphabet="abc", min_size=1, max_size=3))
def test_hypothesis_non_overlapping_conditions(atom: str) -> None:
    parser, checker, metadata = _parser_and_checker()
    token = "".join(ch for ch in atom if ch.isalpha()) or "x"
    condition = f"IF {token}"
    neg_condition = f"UNLESS {token}"

    r1 = _parse(parser, metadata, f"Org MUST collect personal data {condition}.")
    r2 = _parse(parser, metadata, f"Org CANNOT collect personal data {neg_condition}.")

    assert checker.check_conflict(r1, r2) == 0.0
