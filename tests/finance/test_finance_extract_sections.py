from pathlib import Path

import pytest

import tools.finance_extract as fx

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    path = FIXTURE_DIR / name
    return path.read_text(encoding="utf-8")


def test_extract_pairs_respects_liquidity_section_coref():
    html = _load("mdna_liquidity.html")
    pairs = fx.extract_pairs_from_html(html, "10-Q")
    assert pairs, "expected at least one pair from liquidity section"
    liquidity_pairs = [pair for pair in pairs if "Such covenant may be waived" in pair.permission]
    assert liquidity_pairs, "coreference-driven permission should be paired"
    pair = liquidity_pairs[0]
    assert pair.section is not None
    assert pair.section.canonical == "LIQUIDITY"
    assert pair.metadata.get("clause_linked") is True
    assert "ITEM 2" in (pair.section.path or pair.section.title).upper()


def test_credit_agreement_section_boosts_weights():
    html = _load("credit_agreement.html")
    pairs = fx.extract_pairs_from_html(html, "Credit Agreement")
    assert pairs
    pair = pairs[0]
    assert pair.section is not None
    assert pair.section.canonical == "CREDIT_AGREEMENT"
    baseline_ab = fx._authority_balance(f"{pair.obligation} {pair.permission}")
    section_ab = fx._authority_balance(f"{pair.obligation} {pair.permission}", pair.section)
    assert section_ab >= baseline_ab
    base_temporal = fx._temporal_weight("Credit Agreement", None, 90)
    section_temporal = fx._temporal_weight("Credit Agreement", None, 90, pair.section)
    assert section_temporal >= base_temporal


@pytest.mark.parametrize(
    "form, expected_kind",
    [("10-Q", "10-Q"), ("Credit Agreement", "CREDIT_AGREEMENT"), ("10-K", "10-K")],
)
def test_form_to_doc_kind_matches_known_forms(form: str, expected_kind: str):
    doc = fx._build_doc_from_html("<html><body><p>Sample</p></body></html>", form)
    assert doc.doc_kind == expected_kind
