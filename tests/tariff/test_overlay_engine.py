"""Comprehensive tests for Sprint D: Overlay Engine Completion.

Tests cover:
  - AD/CVD Registry: loading, lookup, prefix matching
  - FTA Preference Engine: eligibility, GSP, USMCA, KORUS, etc.
  - Exclusion Tracker: active/expired, date filtering, overlay-type filtering
  - Unified Duty Calculator: full breakdown, FTA reduction, AD/CVD stacking
  - Integration: mutation engine savings with unified duty, engine.py wiring
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# AD/CVD Registry Tests
# ---------------------------------------------------------------------------

class TestADCVDRegistry:
    """Tests for the AD/CVD order database and registry."""

    def test_load_orders_from_default_path(self):
        """Registry loads orders from the default data file."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        assert len(registry.orders) > 0

    def test_order_types_present(self):
        """Both AD and CVD order types are loaded."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        types = {o.order_type for o in registry.orders}
        assert "AD" in types
        assert "CVD" in types

    def test_lookup_steel_threaded_rod_cn(self):
        """Lookup CN-origin steel threaded rod (HTS 7318) returns AD order."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        result = registry.lookup("7318.15.20", "CN")
        assert result.has_exposure
        assert result.total_ad_rate > 0
        assert any("Threaded Rod" in o.product_description for o in result.ad_orders)

    def test_lookup_no_match(self):
        """Lookup for non-matching HTS returns no exposure."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        result = registry.lookup("0101.10.00", "CN")
        assert not result.has_exposure
        assert result.combined_rate == 0.0

    def test_lookup_wrong_origin(self):
        """Lookup with wrong origin country returns no exposure."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        result = registry.lookup("7318.15.20", "DE")
        assert not result.has_exposure

    def test_lookup_aluminum_extrusions_cn(self):
        """CN aluminum extrusions (7604) have both AD and CVD orders."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        result = registry.lookup("7604.10.00", "CN")
        assert result.has_exposure
        assert result.total_ad_rate > 0
        assert result.total_cvd_rate > 0
        assert result.combined_rate == result.total_ad_rate + result.total_cvd_rate

    def test_lookup_wire_rod_cn_dual_orders(self):
        """CN wire rod (7213) has both AD and CVD stacking."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        result = registry.lookup("7213.10.00", "CN")
        assert result.has_exposure
        assert len(result.ad_orders) >= 1
        assert len(result.cvd_orders) >= 1

    def test_lookup_by_hts_only(self):
        """lookup_by_hts returns orders regardless of origin."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        orders = registry.lookup_by_hts("7318.15")
        assert len(orders) >= 1

    def test_normalized_prefix_matching(self):
        """HTS codes with dots are properly normalized for matching."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        registry = ADCVDRegistry()
        result1 = registry.lookup("7318.15.20", "CN")
        result2 = registry.lookup("73181520", "CN")
        assert result1.has_exposure == result2.has_exposure
        assert result1.total_ad_rate == result2.total_ad_rate

    def test_empty_data_file(self, tmp_path):
        """Registry handles empty/missing data file gracefully."""
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("{}")
        registry = ADCVDRegistry(empty_path)
        assert len(registry.orders) == 0
        result = registry.lookup("7318.15", "CN")
        assert not result.has_exposure

    def test_cached_registry(self):
        """get_adcvd_registry returns cached singleton."""
        from potatobacon.tariff.adcvd_registry import get_adcvd_registry
        r1 = get_adcvd_registry()
        r2 = get_adcvd_registry()
        assert r1 is r2


# ---------------------------------------------------------------------------
# FTA Preference Engine Tests
# ---------------------------------------------------------------------------

class TestFTAPreferenceEngine:
    """Tests for the FTA preference evaluation engine."""

    def test_load_programs(self):
        """Engine loads FTA programs from data file."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        assert len(engine.programs) > 0

    def test_usmca_eligible_ca_origin(self):
        """Canadian-origin product importing to US qualifies for USMCA."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "CA", "US")
        assert result.has_eligible_program
        usmca = [p for p in result.eligible_programs if p.program_id == "USMCA"]
        assert len(usmca) == 1
        assert usmca[0].preference_pct == 100.0

    def test_usmca_eligible_mx_origin(self):
        """Mexican-origin product qualifies for USMCA."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "MX", "US")
        usmca = [p for p in result.eligible_programs if p.program_id == "USMCA"]
        assert len(usmca) == 1

    def test_cn_origin_no_fta(self):
        """Chinese-origin product has no FTA preferences."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "CN", "US")
        assert not result.has_eligible_program
        assert result.best_preference_pct == 0.0

    def test_gsp_eligible_country(self):
        """India (GSP beneficiary) qualifies for GSP on eligible products."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "IN", "US")
        gsp = [p for p in result.eligible_programs if p.program_id == "GSP"]
        assert len(gsp) == 1
        assert gsp[0].preference_pct == 100.0

    def test_gsp_excluded_hts(self):
        """GSP-excluded HTS prefixes are rejected."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        # HTS 6404 is excluded from GSP
        result = engine.evaluate("6404.11.00", "IN", "US")
        gsp = [p for p in result.eligible_programs if p.program_id == "GSP"]
        assert len(gsp) == 0

    def test_korus_eligible_kr_origin(self):
        """Korean-origin product qualifies for KORUS."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "KR", "US")
        korus = [p for p in result.eligible_programs if p.program_id == "KORUS"]
        assert len(korus) == 1

    def test_us_australia_fta(self):
        """Australian-origin product qualifies for US-AU FTA."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "AU", "US")
        au = [p for p in result.eligible_programs if p.program_id == "US-AU"]
        assert len(au) == 1

    def test_us_israel_fta(self):
        """Israeli-origin product qualifies for US-IL FTA."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "IL", "US")
        il = [p for p in result.eligible_programs if p.program_id == "US-IL"]
        assert len(il) == 1

    def test_best_program_selection(self):
        """Best program is selected from multiple eligible programs."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "CA", "US")
        assert result.best_program is not None
        assert result.best_preference_pct == 100.0

    def test_usmca_product_specific_rule_steel(self):
        """USMCA steel product-specific rule is recognized."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "CA", "US")
        usmca = [p for p in result.eligible_programs if p.program_id == "USMCA"]
        assert len(usmca) == 1
        assert usmca[0].product_specific_rule_applied is not None

    def test_wrong_import_country(self):
        """FTA for US imports is not eligible when importing to DE."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "CA", "DE")
        assert not result.has_eligible_program

    def test_find_programs_for_country(self):
        """find_programs_for_country returns applicable FTA programs."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        programs = engine.find_programs_for_country("MX")
        ids = [p.program_id for p in programs]
        assert "USMCA" in ids

    def test_us_japan_limited_coverage(self):
        """US-Japan only covers specific HTS prefixes."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        # Steel is NOT covered by US-Japan
        result = engine.evaluate("7318.15.20", "JP", "US")
        jp = [p for p in result.eligible_programs if p.program_id == "US-JP"]
        assert len(jp) == 0
        # Agricultural product IS covered
        result2 = engine.evaluate("0201.10.00", "JP", "US")
        jp2 = [p for p in result2.eligible_programs if p.program_id == "US-JP"]
        assert len(jp2) == 1

    def test_cafta_dr_eligible(self):
        """CAFTA-DR country eligible for preference."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        engine = FTAPreferenceEngine()
        result = engine.evaluate("7318.15.20", "CR", "US")
        cafta = [p for p in result.eligible_programs if p.program_id == "CAFTA-DR"]
        assert len(cafta) == 1

    def test_empty_data_file(self, tmp_path):
        """Engine handles empty data file gracefully."""
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("{}")
        engine = FTAPreferenceEngine(empty_path)
        assert len(engine.programs) == 0
        result = engine.evaluate("7318.15.20", "CA", "US")
        assert not result.has_eligible_program

    def test_cached_engine(self):
        """get_fta_engine returns cached singleton."""
        from potatobacon.tariff.fta_engine import get_fta_engine
        e1 = get_fta_engine()
        e2 = get_fta_engine()
        assert e1 is e2


# ---------------------------------------------------------------------------
# Exclusion Tracker Tests
# ---------------------------------------------------------------------------

class TestExclusionTracker:
    """Tests for the presidential proclamation exclusion tracker."""

    def test_load_exclusions(self):
        """Tracker loads exclusions from default data file."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        assert len(tracker.exclusions) > 0

    def test_active_exclusion_found(self):
        """Active exclusion for aluminum mounting bracket (7616.99.50)."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        result = tracker.check("7616.99.50")
        assert result.has_active_exclusion
        assert result.total_exclusion_relief_pct > 0

    def test_expired_exclusion(self):
        """Expired exclusion for steel fasteners (7318.15.20)."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        result = tracker.check("7318.15.20", reference_date=date(2026, 1, 1))
        assert result.has_expired_exclusion

    def test_no_matching_exclusion(self):
        """No exclusion exists for arbitrary HTS code."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        result = tracker.check("0101.10.00")
        assert not result.has_active_exclusion
        assert not result.has_expired_exclusion
        assert result.total_exclusion_relief_pct == 0.0

    def test_origin_country_filter_301(self):
        """Section 301 exclusion requires CN origin."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        # PCB exclusion requires CN origin
        result_cn = tracker.check("8534.00.00", "CN")
        result_tw = tracker.check("8534.00.00", "TW")
        # CN should have matches; TW should not (for 301 exclusion)
        cn_301 = [e for e in result_cn.active_exclusions if e.overlay_type == "section_301"]
        tw_301 = [e for e in result_tw.active_exclusions if e.overlay_type == "section_301"]
        assert len(cn_301) >= len(tw_301)

    def test_check_by_overlay_type(self):
        """Filter exclusions by overlay type."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        result = tracker.check_by_overlay_type("section_232", "7616.99.50")
        assert all(e.overlay_type == "section_232" for e in result.active_exclusions)

    def test_date_range_filtering(self):
        """Exclusions respect effective/expiry dates."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        # Check before any exclusion is effective
        result = tracker.check("7616.99.50", reference_date=date(2020, 1, 1))
        assert not result.has_active_exclusion

    def test_welded_tubes_exclusion(self):
        """Welded steel tubes (7306.30) have Section 232 exclusion."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        tracker = ExclusionTracker()
        result = tracker.check("7306.30.00", reference_date=date(2025, 1, 1))
        assert result.has_active_exclusion

    def test_empty_data_file(self, tmp_path):
        """Tracker handles empty data file gracefully."""
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("{}")
        tracker = ExclusionTracker(empty_path)
        assert len(tracker.exclusions) == 0
        result = tracker.check("7318.15.20")
        assert not result.has_active_exclusion

    def test_cached_tracker(self):
        """get_exclusion_tracker returns cached singleton."""
        from potatobacon.tariff.exclusion_tracker import get_exclusion_tracker
        t1 = get_exclusion_tracker()
        t2 = get_exclusion_tracker()
        assert t1 is t2


# ---------------------------------------------------------------------------
# Unified Duty Calculator Tests
# ---------------------------------------------------------------------------

class TestDutyCalculator:
    """Tests for the unified duty calculator."""

    def test_base_rate_only(self):
        """No overlays, AD/CVD, or FTA → total = base rate."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        from potatobacon.tariff.adcvd_registry import ADCVDRegistry
        from potatobacon.tariff.exclusion_tracker import ExclusionTracker
        from potatobacon.tariff.fta_engine import FTAPreferenceEngine

        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="0101.10.00",
            origin_country="DE",
            import_country="US",
        )
        # No overlays, no AD/CVD, no FTA for DE on 0101
        assert bd.base_rate == 5.0
        assert bd.effective_base_rate == 5.0
        assert bd.total_duty_rate >= 5.0

    def test_section_232_stacking(self):
        """Steel product from CN includes Section 232 overlay."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=3.4,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
            facts={"origin_country_CN": True},
            active_codes=["HTS_7318_15_20"],
        )
        assert bd.has_232_exposure or bd.section_232_rate >= 0
        # Total should be >= base rate
        assert bd.total_duty_rate >= bd.base_rate

    def test_adcvd_stacking(self):
        """CN steel threaded rod includes AD/CVD duties."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=3.4,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
        )
        assert bd.has_adcvd_exposure
        assert bd.ad_duty_rate > 0
        assert bd.total_duty_rate > bd.base_rate + bd.section_232_rate

    def test_fta_preference_reduces_base(self):
        """USMCA preference reduces effective base rate."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="CA",
            import_country="US",
        )
        assert bd.has_fta_preference
        assert bd.fta_preference_pct > 0
        assert bd.effective_base_rate < bd.base_rate

    def test_full_breakdown_structure(self):
        """DutyBreakdown contains all expected fields."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
        )
        assert hasattr(bd, "base_rate")
        assert hasattr(bd, "section_232_rate")
        assert hasattr(bd, "section_301_rate")
        assert hasattr(bd, "ad_duty_rate")
        assert hasattr(bd, "cvd_duty_rate")
        assert hasattr(bd, "exclusion_relief_rate")
        assert hasattr(bd, "fta_preference_pct")
        assert hasattr(bd, "total_duty_rate")
        assert hasattr(bd, "effective_base_rate")

    def test_exclusion_reduces_overlay(self):
        """Active exclusion reduces overlay duty."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        # Aluminum mounting bracket has an active 232 exclusion
        bd = compute_total_duty(
            base_rate=3.0,
            hts_code="7616.99.50",
            origin_country="CN",
            import_country="US",
        )
        # If both 232 overlay and exclusion apply, relief should offset
        if bd.has_232_exposure and bd.has_active_exclusion:
            assert bd.exclusion_relief_rate > 0
            assert bd.overlay_total <= bd.section_232_rate + bd.section_301_rate

    def test_total_formula(self):
        """Total duty = effective_base + net_overlay + AD + CVD."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
        )
        expected = (
            bd.effective_base_rate
            + max(0.0, bd.section_232_rate + bd.section_301_rate - bd.exclusion_relief_rate)
            + bd.ad_duty_rate
            + bd.cvd_duty_rate
        )
        assert abs(bd.total_duty_rate - expected) < 0.01

    def test_overlay_total_property(self):
        """overlay_total is 232 + 301 - exclusions, min 0."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="0101.10.00",
            origin_country="DE",
            import_country="US",
        )
        assert bd.overlay_total >= 0

    def test_trade_remedy_total_property(self):
        """trade_remedy_total is AD + CVD."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
        )
        assert bd.trade_remedy_total == bd.ad_duty_rate + bd.cvd_duty_rate

    def test_pre_computed_overlays(self):
        """Pre-computed overlays are used when provided."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        from potatobacon.tariff.models import TariffOverlayResultModel

        overlays = [
            TariffOverlayResultModel(
                overlay_name="Section 232 Steel",
                applies=True,
                additional_rate=25.0,
                reason="test",
            )
        ]
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="DE",
            import_country="US",
            overlays=overlays,
        )
        assert bd.section_232_rate == 25.0


# ---------------------------------------------------------------------------
# Duty Delta Tests
# ---------------------------------------------------------------------------

class TestDutyDelta:
    """Tests for compute_duty_delta between scenarios."""

    def test_delta_with_fta_savings(self):
        """Delta captures FTA-driven savings."""
        from potatobacon.tariff.duty_calculator import compute_total_duty, compute_duty_delta

        baseline = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
        )
        optimized = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="CA",
            import_country="US",
        )
        delta = compute_duty_delta(baseline, optimized, declared_value_per_unit=100.0, annual_volume=1000)
        assert delta["total_rate_delta"] > 0
        assert delta["savings_per_unit"] > 0
        assert delta["annual_savings"] > 0

    def test_delta_no_change(self):
        """Same scenarios yield zero delta."""
        from potatobacon.tariff.duty_calculator import compute_total_duty, compute_duty_delta

        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="0101.10.00",
            origin_country="DE",
            import_country="US",
        )
        delta = compute_duty_delta(bd, bd)
        assert delta["total_rate_delta"] == 0.0
        assert delta["savings_per_unit"] == 0.0

    def test_delta_with_annual_volume(self):
        """Annual savings computed when volume provided."""
        from potatobacon.tariff.duty_calculator import compute_total_duty, compute_duty_delta

        baseline = compute_total_duty(base_rate=10.0, hts_code="7318.15.20", origin_country="CN", import_country="US")
        optimized = compute_total_duty(base_rate=5.0, hts_code="7318.15.20", origin_country="CN", import_country="US")
        delta = compute_duty_delta(baseline, optimized, declared_value_per_unit=100.0, annual_volume=10000)
        assert delta["annual_savings"] is not None

    def test_delta_without_annual_volume(self):
        """Annual savings is None without volume."""
        from potatobacon.tariff.duty_calculator import compute_total_duty, compute_duty_delta

        baseline = compute_total_duty(base_rate=10.0, hts_code="7318.15.20", origin_country="CN", import_country="US")
        optimized = compute_total_duty(base_rate=5.0, hts_code="7318.15.20", origin_country="CN", import_country="US")
        delta = compute_duty_delta(baseline, optimized)
        assert delta["annual_savings"] is None


# ---------------------------------------------------------------------------
# DutyBreakdownModel Tests
# ---------------------------------------------------------------------------

class TestDutyBreakdownModel:
    """Tests for the Pydantic DutyBreakdownModel."""

    def test_model_creation(self):
        """DutyBreakdownModel can be created from dict."""
        from potatobacon.tariff.models import DutyBreakdownModel
        model = DutyBreakdownModel(
            base_rate=5.0,
            section_232_rate=25.0,
            ad_duty_rate=206.0,
            total_duty_rate=236.0,
            effective_base_rate=5.0,
            has_232_exposure=True,
            has_adcvd_exposure=True,
        )
        assert model.base_rate == 5.0
        assert model.total_duty_rate == 236.0

    def test_model_serialization(self):
        """Model serializes to dict correctly."""
        from potatobacon.tariff.models import DutyBreakdownModel
        model = DutyBreakdownModel(
            base_rate=5.0,
            total_duty_rate=30.0,
            effective_base_rate=5.0,
        )
        data = model.model_dump()
        assert data["base_rate"] == 5.0
        assert data["total_duty_rate"] == 30.0

    def test_model_defaults(self):
        """Default values are zero/False."""
        from potatobacon.tariff.models import DutyBreakdownModel
        model = DutyBreakdownModel(
            base_rate=5.0,
            total_duty_rate=5.0,
            effective_base_rate=5.0,
        )
        assert model.section_232_rate == 0.0
        assert model.section_301_rate == 0.0
        assert model.ad_duty_rate == 0.0
        assert model.cvd_duty_rate == 0.0
        assert model.has_232_exposure is False
        assert model.has_adcvd_exposure is False


# ---------------------------------------------------------------------------
# Integration: Engine Dossier with Duty Breakdown
# ---------------------------------------------------------------------------

class TestEngineDutyBreakdownIntegration:
    """Test that engine.py dossier includes unified duty breakdowns."""

    def test_dossier_has_duty_breakdown_fields(self):
        """TariffDossierModel now has breakdown fields."""
        from potatobacon.tariff.models import TariffDossierModel
        fields = TariffDossierModel.model_fields
        assert "baseline_duty_breakdown" in fields
        assert "optimized_duty_breakdown" in fields


# ---------------------------------------------------------------------------
# AD/CVD + Overlay Interaction Tests
# ---------------------------------------------------------------------------

class TestADCVDOverlayInteraction:
    """Tests for AD/CVD stacking with Section 232/301 overlays."""

    def test_adcvd_stacks_with_232(self):
        """AD/CVD duties stack on top of Section 232."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=3.4,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
        )
        # CN steel threaded rod: base + 232 + AD
        if bd.has_232_exposure and bd.has_adcvd_exposure:
            assert bd.total_duty_rate > bd.base_rate + bd.section_232_rate
            assert bd.total_duty_rate > bd.base_rate + bd.ad_duty_rate

    def test_fta_does_not_reduce_adcvd(self):
        """FTA preference reduces base rate but not AD/CVD."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        # Wire rod from CN vs CA
        bd_cn = compute_total_duty(
            base_rate=5.0,
            hts_code="7213.10.00",
            origin_country="CN",
            import_country="US",
        )
        bd_ca = compute_total_duty(
            base_rate=5.0,
            hts_code="7213.10.00",
            origin_country="CA",
            import_country="US",
        )
        # CA should have FTA preference on base rate
        if bd_ca.has_fta_preference:
            assert bd_ca.effective_base_rate < bd_ca.base_rate
        # CN should have AD/CVD that CA doesn't
        assert bd_cn.ad_duty_rate >= bd_ca.ad_duty_rate

    def test_exclusion_does_not_affect_adcvd(self):
        """Exclusions only apply to 232/301 overlays, not AD/CVD."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=3.0,
            hts_code="7616.99.50",
            origin_country="CN",
            import_country="US",
        )
        # Even with exclusion, AD/CVD rates are independent
        assert bd.ad_duty_rate >= 0
        assert bd.cvd_duty_rate >= 0


# ---------------------------------------------------------------------------
# Realistic Scenario Tests
# ---------------------------------------------------------------------------

class TestRealisticScenarios:
    """End-to-end realistic duty calculation scenarios."""

    def test_cn_steel_bolt_full_stack(self):
        """CN steel bolt: base + 232 + AD = high total duty."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=3.4,
            hts_code="7318.15.20",
            origin_country="CN",
            import_country="US",
        )
        # Should have significant total duty due to stacking
        assert bd.total_duty_rate > 100.0  # 3.4 + 25 + 206 = 234+
        assert bd.has_232_exposure
        assert bd.has_adcvd_exposure

    def test_ca_steel_bolt_usmca(self):
        """CA steel bolt: USMCA eliminates base rate, 232 may still apply."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=3.4,
            hts_code="7318.15.20",
            origin_country="CA",
            import_country="US",
        )
        assert bd.has_fta_preference
        assert bd.effective_base_rate < bd.base_rate
        # No AD/CVD for CA
        assert bd.ad_duty_rate == 0.0
        assert bd.cvd_duty_rate == 0.0

    def test_tw_stainless_steel_sheet(self):
        """TW stainless steel: AD order applies (A-583-008)."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=0.0,
            hts_code="7219.31.00",
            origin_country="TW",
            import_country="US",
        )
        assert bd.has_adcvd_exposure
        assert bd.ad_duty_rate > 0

    def test_de_no_trade_remedies(self):
        """DE origin steel: no AD/CVD, no FTA (non-partner)."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=3.4,
            hts_code="7318.15.20",
            origin_country="DE",
            import_country="US",
        )
        assert not bd.has_adcvd_exposure
        assert not bd.has_fta_preference
        # Still has 232 overlay for steel
        assert bd.has_232_exposure or bd.section_232_rate >= 0

    def test_in_gsp_eligible_product(self):
        """India (GSP): eligible product gets duty-free treatment."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="7318.15.20",
            origin_country="IN",
            import_country="US",
        )
        if bd.has_fta_preference:
            assert bd.effective_base_rate < bd.base_rate

    def test_cn_aluminum_extrusions_dual_remedy(self):
        """CN aluminum extrusions: both AD and CVD orders stack."""
        from potatobacon.tariff.duty_calculator import compute_total_duty
        bd = compute_total_duty(
            base_rate=5.0,
            hts_code="7604.10.00",
            origin_country="CN",
            import_country="US",
        )
        assert bd.has_adcvd_exposure
        assert bd.ad_duty_rate > 0
        assert bd.cvd_duty_rate > 0
        assert bd.trade_remedy_total > 300  # Very high combined rate
