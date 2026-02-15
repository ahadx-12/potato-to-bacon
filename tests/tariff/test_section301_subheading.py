"""Tests for Section 301 at subheading level with heading fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from potatobacon.tariff.overlays import evaluate_overlays
from potatobacon.tariff.models import TariffOverlayResultModel


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Create a temp overlay directory with both subheading and heading files."""
    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()

    # Subheading-level 301 data (8-digit)
    subheading = {
        "overlays": [
            {
                "overlay_name": "Section 301 - List 1 (25%)",
                "hts_prefixes": ["8471.30.01", "8471.41.01", "8517.62.00"],
                "additional_rate": 0.25,
                "reason": "Section 301 List 1",
                "stop_optimization": True,
                "origin_countries": ["CN"],
                "match_level": "exact_8digit",
            }
        ]
    }
    (overlay_dir / "section301_subheading.json").write_text(
        json.dumps(subheading), encoding="utf-8"
    )

    # Heading-level 301 data (4-digit — some overlap with subheading)
    heading = {
        "overlays": [
            {
                "overlay_name": "Section 301 - List 1 (25%)",
                "hts_prefixes": ["8471", "8517", "9903"],
                "additional_rate": 0.25,
                "reason": "Section 301 heading level",
                "stop_optimization": True,
                "origin_countries": ["CN"],
            }
        ]
    }
    (overlay_dir / "section301_sample.json").write_text(
        json.dumps(heading), encoding="utf-8"
    )

    # Section 232 (always loaded)
    s232 = {
        "overlays": [
            {
                "overlay_name": "Section 232 - Steel (25%)",
                "hts_prefixes": ["7206", "7207", "7208"],
                "additional_rate": 0.25,
                "reason": "Section 232 steel tariff",
                "stop_optimization": True,
                "origin_countries": [],
            }
        ]
    }
    (overlay_dir / "section232_sample.json").write_text(
        json.dumps(s232), encoding="utf-8"
    )

    return overlay_dir


class TestSubheadingMatching:
    """Test that 8-digit matching takes priority over heading matching."""

    def test_exact_8digit_match(self, data_root: Path):
        """An 8-digit HTS code covered by subheading data should match at exact level."""
        overlays = evaluate_overlays(
            facts=None,
            active_codes=None,
            origin_country="CN",
            hts_code="8471.30.01",
            data_root=str(data_root),
        )
        s301 = [o for o in overlays if "301" in o.overlay_name]
        assert len(s301) >= 1
        # At least one should be exact_8digit
        exact = [o for o in s301 if o.match_level == "exact_8digit"]
        assert len(exact) >= 1
        assert exact[0].additional_rate == 0.25

    def test_heading_fallback_for_uncovered_code(self, data_root: Path):
        """9903 heading is NOT covered at 8-digit level, so heading fallback should apply."""
        overlays = evaluate_overlays(
            facts=None,
            active_codes=None,
            origin_country="CN",
            hts_code="9903.88.15",
            data_root=str(data_root),
        )
        s301 = [o for o in overlays if "301" in o.overlay_name]
        # 9903 should match from heading-level fallback
        fallback = [o for o in s301 if o.match_level == "heading_fallback"]
        assert len(fallback) >= 1

    def test_section_232_unaffected(self, data_root: Path):
        """Section 232 loading should be unaffected by subheading changes."""
        overlays = evaluate_overlays(
            facts=None,
            active_codes=None,
            origin_country="CN",
            hts_code="7208.10.00",
            data_root=str(data_root),
        )
        s232 = [o for o in overlays if "232" in o.overlay_name]
        assert len(s232) >= 1
        assert s232[0].additional_rate == 0.25

    def test_non_china_origin_no_match(self, data_root: Path):
        """Section 301 should not apply to non-China origins."""
        overlays = evaluate_overlays(
            facts=None,
            active_codes=None,
            origin_country="DE",
            hts_code="8471.30.01",
            data_root=str(data_root),
        )
        s301 = [o for o in overlays if "301" in o.overlay_name]
        assert len(s301) == 0


class TestMatchLevelField:
    """Test that match_level is properly propagated to TariffOverlayResultModel."""

    def test_match_level_on_model(self):
        """TariffOverlayResultModel should accept match_level field."""
        model = TariffOverlayResultModel(
            overlay_name="Section 301 Test",
            applies=True,
            additional_rate=0.25,
            reason="Test",
            match_level="exact_8digit",
        )
        assert model.match_level == "exact_8digit"

    def test_match_level_defaults_empty(self):
        """match_level should default to empty string."""
        model = TariffOverlayResultModel(
            overlay_name="Section 232 Test",
            applies=True,
            additional_rate=0.25,
            reason="Test",
        )
        assert model.match_level == ""


class TestLiveDataIntegration:
    """Integration tests against actual Section 301 subheading data if available."""

    @pytest.fixture
    def live_data_root(self) -> str | None:
        path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "overlays"
        )
        if (path / "section301_subheading.json").exists():
            return str(path)
        return None

    def test_live_301_list1_technology(self, live_data_root: str | None):
        """8471.30.01 (laptops) should match Section 301 List 1 from live data."""
        if live_data_root is None:
            pytest.skip("Live section301_subheading.json not available")
        overlays = evaluate_overlays(
            facts=None,
            active_codes=None,
            origin_country="CN",
            hts_code="8471.30.01",
            data_root=live_data_root,
        )
        s301 = [o for o in overlays if "301" in o.overlay_name and o.applies]
        assert len(s301) >= 1
        assert any(o.additional_rate == 0.25 for o in s301)

    def test_live_301_list4a_apparel(self, live_data_root: str | None):
        """6109.10.00 (T-shirts) should match Section 301 List 4A at 7.5%."""
        if live_data_root is None:
            pytest.skip("Live section301_subheading.json not available")
        overlays = evaluate_overlays(
            facts=None,
            active_codes=None,
            origin_country="CN",
            hts_code="6109.10.00",
            data_root=live_data_root,
        )
        s301 = [o for o in overlays if "301" in o.overlay_name and "4A" in o.overlay_name]
        assert len(s301) >= 1
        assert any(abs(o.additional_rate - 0.075) < 0.001 for o in s301)
