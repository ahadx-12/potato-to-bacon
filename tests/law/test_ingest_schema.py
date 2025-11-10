from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative",
    [
        "us_tax_corpus/raw/usc_title26_xml",
        "us_tax_corpus/raw/ecfr_title26",
        "us_tax_corpus/raw/irb_html",
        "us_tax_corpus/raw/tax_court_json",
        "us_tax_corpus/parsed",
        "us_tax_corpus/manifests",
        "us_tax_corpus/logs",
        "out",
    ],
)
def test_required_directories_exist(relative: str) -> None:
    path = ROOT / relative
    assert path.exists(), f"Expected {relative} to be scaffolded"
