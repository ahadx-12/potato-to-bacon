import re
from pathlib import Path

from scripts.generate_tariff_readiness_report import generate_report


def test_report_includes_all_categories(tmp_path):
    report_path = generate_report()
    assert report_path.exists(), "Report path should exist"

    content = report_path.read_text(encoding="utf-8")

    for token in ["electronics", "apparel_textile", "unknown", "Category scorecard"]:
        assert re.search(token, content, re.IGNORECASE), f"Missing section for {token}"

    assert "no AD/CVD or origin logic" not in content

    # ensure report file stored in reports directory
    assert Path("reports") in report_path.parents
