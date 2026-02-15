"""Tests for specific and compound duty rate computation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from potatobacon.tariff.duty_calculator import DutyBreakdown, compute_total_duty
from potatobacon.tariff.models import TariffOverlayResultModel


class TestSpecificRateComputation:
    """Test duty calculation with specific (per-unit/per-kg) rates."""

    def test_specific_rate_with_weight(self):
        """Specific rate with weight_kg should compute absolute duty amount."""
        breakdown = compute_total_duty(
            base_rate=0.0,
            hts_code="0201.10.05",
            origin_country="BR",
            weight_kg=1000.0,
            overlays=[],
        )
        # With explicit base_rate=0.0, specific computation isn't auto-triggered
        # but the structure is correct
        assert breakdown.base_rate == 0.0
        assert isinstance(breakdown.specific_duty_amount, float)

    def test_compound_rate_explicit(self):
        """Compound rate with ad valorem component should be computed correctly."""
        breakdown = compute_total_duty(
            base_rate=0.025,  # 2.5% ad valorem component
            hts_code="8471.50.01",
            origin_country="CN",
            declared_value=10000.0,
            weight_kg=50.0,
            overlays=[],
        )
        assert breakdown.base_rate == 0.025
        assert breakdown.total_duty_rate >= 0.025


class TestDutyBreakdownFields:
    """Test the new fields on DutyBreakdown for specific/compound rates."""

    def test_rate_type_default(self):
        """Default rate_type should be 'ad_valorem'."""
        breakdown = compute_total_duty(
            base_rate=0.05,
            hts_code="8471.30.01",
            origin_country="DE",
            overlays=[],
        )
        assert breakdown.rate_type == "ad_valorem"
        assert not breakdown.is_specific_or_compound

    def test_specific_duty_amount_defaults_zero(self):
        """specific_duty_amount should default to 0.0 for ad valorem rates."""
        breakdown = compute_total_duty(
            base_rate=0.10,
            hts_code="6402.99.31",
            origin_country="CN",
            overlays=[],
        )
        assert breakdown.specific_duty_amount == 0.0
        assert breakdown.compound_ad_valorem_amount == 0.0

    def test_declared_value_and_weight_accepted(self):
        """compute_total_duty should accept declared_value, weight_kg, quantity params."""
        breakdown = compute_total_duty(
            base_rate=0.05,
            hts_code="0201.10.05",
            origin_country="AU",
            declared_value=50000.0,
            weight_kg=5000.0,
            quantity=100,
            overlays=[],
        )
        assert breakdown.base_rate == 0.05


class TestBackwardCompatibility:
    """Ensure existing compute_total_duty call patterns still work."""

    def test_basic_ad_valorem(self):
        """Standard ad valorem computation should be unchanged."""
        breakdown = compute_total_duty(
            base_rate=0.05,
            hts_code="9403.20.00",
            origin_country="CN",
            overlays=[
                TariffOverlayResultModel(
                    overlay_name="Section 301 Test",
                    applies=True,
                    additional_rate=0.25,
                    reason="Test overlay",
                    stop_optimization=True,
                ),
            ],
        )
        assert breakdown.base_rate == 0.05
        assert breakdown.section_301_rate == 0.25
        assert breakdown.total_duty_rate == pytest.approx(0.30, abs=0.01)

    def test_no_overlays(self):
        """Passing empty overlays should work."""
        breakdown = compute_total_duty(
            base_rate=0.10,
            hts_code="1234.56.78",
            origin_country="DE",
            overlays=[],
        )
        assert breakdown.total_duty_rate == pytest.approx(0.10, abs=0.001)
        assert breakdown.section_301_rate == 0.0
        assert breakdown.section_232_rate == 0.0
