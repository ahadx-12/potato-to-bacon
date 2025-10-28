import pytest

from potatobacon.cale.finance.docio import Doc, DocBlock
from potatobacon.cale.finance.sectionizer import find_sections


def _make_doc(positives, negatives):
    blocks = []
    metric_text = "The borrower shall maintain a leverage ratio of 3.0 to 1.0 and minimum liquidity of $50 million."
    negative_text = "Management discussion and analysis of results of operations and risk factors."
    for header in positives:
        blocks.append(DocBlock("header", text=header))
        blocks.append(DocBlock("paragraph", text=metric_text))
    for header in negatives:
        blocks.append(DocBlock("header", text=header))
        blocks.append(DocBlock("paragraph", text=negative_text))
    return Doc(src_path="/tmp/doc.html", doc_kind="INDENTURE", blocks=blocks)


POSITIVES = [f"Section 6.{i:02d} Financial Covenant" for i in range(1, 31)]
POSITIVES += [f"Negative Covenant - Restricted Payments {i}" for i in range(10)]
NEGATIVES = [f"Item 7. Management Discussion and Analysis {i}" for i in range(30)]
NEGATIVES += [f"Risk Factors Overview {i}" for i in range(10)]

DOC = _make_doc(POSITIVES, NEGATIVES)
SECTIONS = find_sections(DOC)
TITLES = {section.title for section in SECTIONS}
MIXED_HEADERS = [(POSITIVES[i], True) for i in range(10)] + [
    (NEGATIVES[i], False) for i in range(10)
]


@pytest.mark.parametrize("header", POSITIVES)
def test_positive_headers_are_selected(header):
    assert header in TITLES


@pytest.mark.parametrize("header", NEGATIVES)
def test_negative_headers_are_rejected(header):
    assert header not in TITLES


@pytest.mark.parametrize("header, expected", MIXED_HEADERS)
def test_mixed_header_predictions(header, expected):
    assert (header in TITLES) is expected


def test_positive_negative_detection():
    tp = sum(1 for header in POSITIVES if header in TITLES)
    fp = sum(1 for header in NEGATIVES if header in TITLES)
    fn = len(POSITIVES) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / len(POSITIVES)
    f1 = (2 * precision * recall) / (precision + recall)
    assert f1 >= 0.88


def test_section_windows_are_valid():
    positives = [f"Section 7.{i:02d} Financial Covenant" for i in range(1, 11)]
    negatives = [f"Item 1A. Risk Factors {i}" for i in range(10)]
    doc = _make_doc(positives, negatives)
    sections = find_sections(doc)
    for section in sections:
        assert doc.blocks[section.start_block].kind == "header"
        assert section.end_block > section.start_block
        assert section.score >= 0.9
