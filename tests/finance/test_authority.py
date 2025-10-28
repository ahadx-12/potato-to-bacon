from pathlib import Path

from potatobacon.cale.finance import authority
from potatobacon.cale.finance.docio import load_doc

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sec_small"


def test_html_anchor_resolution():
    doc = load_doc(FIXTURES / "indenture_2020.html")
    anchors = [authority.link_block(doc, idx) for idx in range(len(doc.blocks))]
    resolved = sum(1 for info in anchors if info.get("anchor"))
    assert resolved / len(anchors) >= 0.95
    assert anchors[0]["anchor"] == "art6"
    assert anchors[0]["section_title"].startswith("Article VI")
    assert anchors[2]["anchor"] == "sec602"
    assert anchors[3]["section_title"].startswith("Section 6.02")
    range_info = authority.link_range(doc, 0, 3)
    assert range_info["anchor"] == "art6"


def test_ascii_anchor_generation():
    doc = load_doc(FIXTURES / "credit_agreement_2019.txt")
    anchors = [authority.link_block(doc, idx) for idx in range(len(doc.blocks))]
    resolved = sum(1 for info in anchors if info.get("anchor"))
    assert resolved / len(anchors) >= 0.95
    assert anchors[0]["anchor"].startswith("sec-")
    assert anchors[1]["section_title"].startswith("ARTICLE VI")
