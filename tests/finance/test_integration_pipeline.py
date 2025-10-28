from pathlib import Path

from tools import finance_extract

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sec_small"


def test_offline_pipeline_summary():
    files = [
        str(FIXTURES / "indenture_2020.html"),
        str(FIXTURES / "credit_agreement_2019.txt"),
        str(FIXTURES / "10q_b.html"),
    ]
    summary = finance_extract.run_local_pipeline(files, baseline_pairs=5)
    assert summary["sectionizer"]["pass"] is True
    assert summary["table_parsing"]["pass"] is True
    assert summary["authority_links"]["pass"] is True
    assert summary["numeric_pairs"] >= 8
    assert summary["numeric_pairs"] >= int(summary["baseline_pairs"] * 1.4)
    assert summary["evidence"]
    for row in summary["evidence"]:
        assert "section_key" in row
        assert "section_title" in row
        assert "anchor" in row
        assert "doc_kind" in row
        assert row["qualifiers"]["section"] == row["section_title"]
        if "table_cell_meta" in row:
            meta = row["table_cell_meta"]
            assert set(meta) == {"row_header", "col_header", "row_index", "col_index"}
