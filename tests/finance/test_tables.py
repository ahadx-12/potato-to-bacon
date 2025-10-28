from pathlib import Path

from potatobacon.cale.finance import tables
from potatobacon.cale.finance.docio import load_doc

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sec_small"


def test_html_table_flattening():
    doc = load_doc(FIXTURES / "indenture_2020.html")
    table_block = next(block for block in doc.blocks if block.kind == "table")
    rows = list(tables.flatten(table_block))
    sentences = {row[0] for row in rows}
    expected = {
        "Q1 Maximum Leverage: 3.50x",
        "Q1 Minimum Coverage: 2.0x",
        "Q2 Maximum Leverage: 3.25x",
        "Q2 Minimum Coverage: 2.1x",
    }
    assert sentences == expected
    first_meta = rows[0][1]
    assert first_meta["row_header"] == "Q1"
    assert first_meta["col_header"] == "Maximum Leverage"


def test_ascii_table_flattening_jaccard():
    doc = load_doc(FIXTURES / "credit_agreement_2019.txt")
    table_block = next(block for block in doc.blocks if block.kind == "table")
    rows = list(tables.flatten(table_block))
    output = {row[0] for row in rows}
    expected = {
        "Q3 2019 Max Leverage: 4.00x",
        "Q3 2019 Min Liquidity: $30 million",
        "Q4 2019 Max Leverage: 3.75x",
        "Q4 2019 Min Liquidity: $32 million",
    }
    intersection = len(output & expected)
    union = len(output | expected)
    jaccard = intersection / union
    assert jaccard >= 0.9
