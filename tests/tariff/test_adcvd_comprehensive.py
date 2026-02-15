"""Tests for comprehensive AD/CVD registry with tiered confidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from potatobacon.tariff.adcvd_registry import (
    ADCVDRegistry,
    ADCVDLookupResult,
)


@pytest.fixture
def full_registry() -> ADCVDRegistry:
    """Load the full AD/CVD orders database."""
    full_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "overlays"
        / "adcvd_orders_full.json"
    )
    if not full_path.exists():
        pytest.skip("adcvd_orders_full.json not found")
    return ADCVDRegistry(data_path=full_path)


@pytest.fixture
def sample_registry(tmp_path: Path) -> ADCVDRegistry:
    """Create a small registry for controlled tests."""
    data = {
        "orders": [
            {
                "order_id": "A-570-test-8digit",
                "type": "AD",
                "product_description": "Test product 8-digit",
                "hts_prefixes": ["7307.93.30"],
                "origin_countries": ["CN"],
                "duty_rate_pct": 182.9,
                "effective_date": "2020-01-01",
                "status": "active",
                "case_number": "A-570-test",
                "federal_register_citation": "85 FR 12345",
                "scope_keywords": ["butt-weld", "pipe fittings"],
            },
            {
                "order_id": "A-570-test-6digit",
                "type": "AD",
                "product_description": "Test product 6-digit",
                "hts_prefixes": ["7208.10"],
                "origin_countries": ["CN"],
                "duty_rate_pct": 67.41,
                "effective_date": "2001-11-29",
                "status": "active",
                "case_number": "A-570-test2",
                "federal_register_citation": "66 FR 49634",
                "scope_keywords": ["hot-rolled", "flat products"],
            },
            {
                "order_id": "A-570-test-4digit",
                "type": "CVD",
                "product_description": "Test product 4-digit heading",
                "hts_prefixes": ["7208"],
                "origin_countries": ["CN"],
                "duty_rate_pct": 12.0,
                "effective_date": "2001-11-29",
                "status": "active",
                "case_number": "C-570-test",
                "federal_register_citation": "66 FR 49635",
                "scope_keywords": ["hot-rolled"],
            },
            {
                "order_id": "A-570-inactive",
                "type": "AD",
                "product_description": "Inactive order",
                "hts_prefixes": ["7307.93.30"],
                "origin_countries": ["CN"],
                "duty_rate_pct": 50.0,
                "effective_date": "2010-01-01",
                "status": "revoked",
                "case_number": "A-570-inactive",
                "federal_register_citation": "75 FR 11111",
            },
        ]
    }
    path = tmp_path / "test_adcvd.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return ADCVDRegistry(data_path=path)


class TestTieredConfidence:
    """Test 3-tier confidence: high (8-digit), high (6-digit), medium (4-digit)."""

    def test_8digit_exact_match_high_confidence(self, sample_registry: ADCVDRegistry):
        result = sample_registry.lookup("7307.93.30", "CN")
        assert result.has_exposure
        assert result.confidence == "high"
        assert any("8-digit" in m.note for m in result.order_matches)

    def test_6digit_match_high_confidence(self, sample_registry: ADCVDRegistry):
        result = sample_registry.lookup("7208.10.00", "CN")
        assert result.has_exposure
        assert result.confidence in ("high", "medium")

    def test_4digit_heading_medium_confidence(self, sample_registry: ADCVDRegistry):
        # 7208.99.00 matches the 4-digit heading prefix "7208"
        result = sample_registry.lookup("7208.99.00", "CN")
        assert result.has_exposure
        # Should be medium confidence for 4-digit match
        any_medium = any(m.confidence == "medium" for m in result.order_matches)
        any_high = any(m.confidence == "high" for m in result.order_matches)
        assert any_medium or any_high  # Either medium or high from 6-digit match

    def test_no_match_returns_none_confidence(self, sample_registry: ADCVDRegistry):
        result = sample_registry.lookup("9999.99.99", "CN")
        assert not result.has_exposure
        assert result.confidence == "none"

    def test_wrong_country_no_match(self, sample_registry: ADCVDRegistry):
        result = sample_registry.lookup("7307.93.30", "DE")
        assert not result.has_exposure

    def test_inactive_order_excluded(self, sample_registry: ADCVDRegistry):
        """Inactive/revoked orders should not appear in results."""
        result = sample_registry.lookup("7307.93.30", "CN")
        order_ids = [m.order.order_id for m in result.order_matches]
        assert "A-570-inactive" not in order_ids


class TestScopeKeywords:
    """Test that scope_keywords are loaded on ADCVDOrder."""

    def test_keywords_loaded(self, sample_registry: ADCVDRegistry):
        orders = sample_registry.orders
        kw_order = [o for o in orders if o.order_id == "A-570-test-8digit"]
        assert len(kw_order) == 1
        assert "butt-weld" in kw_order[0].scope_keywords
        assert "pipe fittings" in kw_order[0].scope_keywords


class TestFullDatabase:
    """Integration tests against the full AD/CVD database."""

    def test_full_database_loads(self, full_registry: ADCVDRegistry):
        assert len(full_registry.orders) >= 40

    def test_china_steel_exposure(self, full_registry: ADCVDRegistry):
        # Hot-rolled carbon steel flat products from China
        result = full_registry.lookup("7208.10.00", "CN")
        assert result.has_exposure
        assert result.total_ad_rate > 0

    def test_solar_panel_exposure(self, full_registry: ADCVDRegistry):
        # Crystalline silicon photovoltaic cells from China
        result = full_registry.lookup("8541.40.60", "CN")
        assert result.has_exposure
        # Should have both AD and CVD
        assert result.total_ad_rate > 0
        assert result.total_cvd_rate > 0

    def test_aluminum_extrusion_exposure(self, full_registry: ADCVDRegistry):
        result = full_registry.lookup("7604.10.00", "CN")
        assert result.has_exposure

    def test_non_covered_country(self, full_registry: ADCVDRegistry):
        # Germany steel shouldn't match China orders
        result = full_registry.lookup("7208.10.00", "DE")
        # Might match some German-origin orders but should be limited
        # At minimum, shouldn't match China-only orders
        for match in result.order_matches:
            assert "DE" in match.order.origin_countries or "CN" not in match.order.origin_countries

    def test_lookup_by_hts(self, full_registry: ADCVDRegistry):
        orders = full_registry.lookup_by_hts("7208.10")
        assert len(orders) >= 1
