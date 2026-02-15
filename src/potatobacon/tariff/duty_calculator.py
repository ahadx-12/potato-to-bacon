"""Unified duty calculator.

Computes total landed duty by combining:
  base_rate + section_232 + section_301 + ad_duty + cvd_duty - exclusions - fta_preference

This replaces the scattered duty calculation logic with a single entry point
that accounts for all duty layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from potatobacon.tariff.adcvd_registry import ADCVDLookupResult, ADCVDRegistry, get_adcvd_registry
from potatobacon.tariff.exclusion_tracker import ExclusionLookupResult, ExclusionTracker, get_exclusion_tracker
from potatobacon.tariff.fta_engine import FTALookupResult, FTAPreferenceEngine, get_fta_engine
from potatobacon.tariff.models import TariffOverlayResultModel
from potatobacon.tariff.overlays import effective_duty_rate, evaluate_overlays

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DutyBreakdown:
    """Full breakdown of all duty components for a single scenario."""

    base_rate: float
    section_232_rate: float
    section_301_rate: float
    ad_duty_rate: float
    cvd_duty_rate: float
    exclusion_relief_rate: float
    fta_preference_pct: float  # 0-100; 100 = full elimination of base rate
    total_duty_rate: float
    effective_base_rate: float  # base_rate adjusted by FTA preference

    # Specific/compound rate components
    specific_duty_amount: float = 0.0  # Absolute $ amount from specific rates
    compound_ad_valorem_amount: float = 0.0  # Ad valorem $ amount (for compound)
    is_specific_or_compound: bool = False
    rate_type: str = "ad_valorem"  # ad_valorem | specific | compound | compound_max | free

    # Component details
    overlays: list[TariffOverlayResultModel] = field(default_factory=list)
    adcvd_result: ADCVDLookupResult | None = None
    exclusion_result: ExclusionLookupResult | None = None
    fta_result: FTALookupResult | None = None

    # Flags
    has_232_exposure: bool = False
    has_301_exposure: bool = False
    has_adcvd_exposure: bool = False
    has_fta_preference: bool = False
    has_active_exclusion: bool = False
    stop_optimization: bool = False

    # Rate resolution metadata
    base_rate_source: str = "manual"
    base_rate_warning: Optional[str] = None

    @property
    def overlay_total(self) -> float:
        """Sum of Section 232 + 301 overlay rates, net of exclusions."""
        return max(0.0, self.section_232_rate + self.section_301_rate - self.exclusion_relief_rate)

    @property
    def trade_remedy_total(self) -> float:
        """Sum of AD + CVD rates."""
        return self.ad_duty_rate + self.cvd_duty_rate

    @property
    def savings_vs_no_optimization(self) -> float:
        """Total reduction from FTA preferences and exclusions."""
        return self.exclusion_relief_rate + (
            self.base_rate * self.fta_preference_pct / 100.0 if self.fta_preference_pct > 0 else 0.0
        )


def compute_total_duty(
    *,
    base_rate: Optional[float] = None,
    hts_code: str = "",
    origin_country: str = "",
    import_country: str = "US",
    declared_value: float | None = None,
    weight_kg: float | None = None,
    quantity: int | None = None,
    facts: Mapping[str, Any] | None = None,
    active_codes: Sequence[str] | None = None,
    overlays: list[TariffOverlayResultModel] | None = None,
    adcvd_registry: ADCVDRegistry | None = None,
    exclusion_tracker: ExclusionTracker | None = None,
    fta_engine: FTAPreferenceEngine | None = None,
) -> DutyBreakdown:
    """Compute total landed duty including all overlay layers.

    Parameters
    ----------
    base_rate:
        The base ad-valorem duty rate from HTS classification.
        When *None*, the rate is auto-resolved from the MFN rate store
        using *hts_code*.
    hts_code:
        Full or partial HTS code for overlay matching.
    origin_country:
        ISO 2-letter country of origin.
    import_country:
        ISO 2-letter importing country (default US).
    facts:
        Compiled facts dict for overlay and FTA evaluation.
    active_codes:
        Active HTS atom source IDs from Z3 evaluation.
    overlays:
        Pre-computed overlay results (if None, will be evaluated).
    adcvd_registry:
        Optional custom AD/CVD registry instance.
    exclusion_tracker:
        Optional custom exclusion tracker instance.
    fta_engine:
        Optional custom FTA engine instance.
    """
    facts = facts or {}
    active_codes = list(active_codes or [])

    # --- Auto-resolve base rate from rate store when not provided ---
    base_rate_source = "manual"
    base_rate_warning: Optional[str] = None
    specific_duty_amount = 0.0
    compound_ad_valorem_amount = 0.0
    is_specific_or_compound = False
    rate_type = "ad_valorem"

    if base_rate is None:
        from potatobacon.tariff.rate_store import get_rate_store

        rate_store = get_rate_store()
        lookup = rate_store.lookup(hts_code)
        if lookup.found and lookup.ad_valorem_rate is not None:
            base_rate = lookup.ad_valorem_rate
            base_rate_source = f"auto:{lookup.match_level}"
            if lookup.warning:
                base_rate_warning = lookup.warning
            logger.debug(
                "Auto-resolved base rate for %s: %.2f%% (%s)",
                hts_code, base_rate, lookup.match_level,
            )
        elif lookup.found and lookup.parsed_rate is not None:
            # Handle specific-only and compound rates
            parsed = lookup.parsed_rate
            if parsed.specific_amount is not None and parsed.ad_valorem_pct is None:
                # Specific rate only — compute from weight/quantity
                base_rate = 0.0  # No ad valorem component
                rate_type = "specific"
                is_specific_or_compound = True
                base_rate_source = f"auto:{lookup.match_level}"
                if weight_kg is not None:
                    specific_duty_amount = parsed.specific_amount * weight_kg
                elif quantity is not None:
                    specific_duty_amount = parsed.specific_amount * quantity
                else:
                    base_rate_warning = (
                        f"Specific rate {parsed.raw} requires weight_kg or quantity "
                        "to compute duty amount"
                    )
            elif parsed.is_compound:
                # Compound rate: ad valorem + specific
                base_rate = parsed.ad_valorem_pct / 100.0 if parsed.ad_valorem_pct else 0.0
                rate_type = "compound"
                is_specific_or_compound = True
                base_rate_source = f"auto:{lookup.match_level}"
                if parsed.specific_amount is not None:
                    if weight_kg is not None:
                        specific_duty_amount = parsed.specific_amount * weight_kg
                    elif quantity is not None:
                        specific_duty_amount = parsed.specific_amount * quantity
                if declared_value is not None and base_rate > 0:
                    compound_ad_valorem_amount = base_rate * declared_value
                if lookup.warning:
                    base_rate_warning = lookup.warning
            else:
                base_rate = 0.0
                base_rate_source = "manual_review_required"
                base_rate_warning = lookup.warning or f"Rate for {hts_code} requires manual review"
                logger.warning("Rate for %s requires manual review: %s", hts_code, lookup.warning)
        elif lookup.found and lookup.requires_manual_review:
            base_rate = 0.0
            base_rate_source = "manual_review_required"
            base_rate_warning = lookup.warning or f"Rate for {hts_code} requires manual review"
            logger.warning("Rate for %s requires manual review: %s", hts_code, lookup.warning)
        else:
            raise ValueError(
                f"HTS code {hts_code!r} not found in rate store and no base_rate supplied. "
                "Provide base_rate explicitly or ensure the rate store is populated."
            )

    # --- Section 232/301 overlays ---
    if overlays is None:
        overlays = evaluate_overlays(
            facts=facts,
            active_codes=active_codes,
            origin_country=origin_country or None,
            import_country=import_country or None,
            hts_code=hts_code or None,
        )

    section_232_rate = 0.0
    section_301_rate = 0.0
    stop_opt = False
    for ov in overlays:
        if not ov.applies:
            continue
        name_lower = ov.overlay_name.lower()
        if "232" in name_lower:
            section_232_rate += ov.additional_rate
        elif "301" in name_lower:
            section_301_rate += ov.additional_rate
        if ov.stop_optimization:
            stop_opt = True

    # --- AD/CVD orders ---
    registry = adcvd_registry or get_adcvd_registry()
    adcvd_result: ADCVDLookupResult | None = None
    ad_duty_rate = 0.0
    cvd_duty_rate = 0.0
    if hts_code and origin_country:
        adcvd_result = registry.lookup(hts_code, origin_country)
        ad_duty_rate = adcvd_result.total_ad_rate
        cvd_duty_rate = adcvd_result.total_cvd_rate

    # --- Exclusions ---
    tracker = exclusion_tracker or get_exclusion_tracker()
    exclusion_result: ExclusionLookupResult | None = None
    exclusion_relief = 0.0
    if hts_code:
        exclusion_result = tracker.check(hts_code, origin_country or None)
        exclusion_relief = exclusion_result.total_exclusion_relief_pct

    # Cap exclusion relief to the total overlay it applies against
    overlay_total_before_exclusion = section_232_rate + section_301_rate
    exclusion_relief = min(exclusion_relief, overlay_total_before_exclusion)

    # --- FTA preference ---
    fta = fta_engine or get_fta_engine()
    fta_result: FTALookupResult | None = None
    fta_preference_pct = 0.0
    if hts_code and origin_country and import_country:
        fta_result = fta.evaluate(hts_code, origin_country, import_country, facts)
        fta_preference_pct = fta_result.best_preference_pct

    # --- Compute effective base rate after FTA preference ---
    effective_base = base_rate
    if fta_preference_pct > 0:
        effective_base = base_rate * (1.0 - fta_preference_pct / 100.0)

    # --- Total duty ---
    # total = effective_base + (232 + 301 - exclusions) + AD + CVD
    net_overlay = max(0.0, section_232_rate + section_301_rate - exclusion_relief)
    total = effective_base + net_overlay + ad_duty_rate + cvd_duty_rate

    return DutyBreakdown(
        base_rate=base_rate,
        section_232_rate=section_232_rate,
        section_301_rate=section_301_rate,
        ad_duty_rate=ad_duty_rate,
        cvd_duty_rate=cvd_duty_rate,
        exclusion_relief_rate=exclusion_relief,
        fta_preference_pct=fta_preference_pct,
        total_duty_rate=total,
        effective_base_rate=effective_base,
        specific_duty_amount=specific_duty_amount,
        compound_ad_valorem_amount=compound_ad_valorem_amount,
        is_specific_or_compound=is_specific_or_compound,
        rate_type=rate_type,
        overlays=list(overlays),
        adcvd_result=adcvd_result,
        exclusion_result=exclusion_result,
        fta_result=fta_result,
        has_232_exposure=section_232_rate > 0,
        has_301_exposure=section_301_rate > 0,
        has_adcvd_exposure=adcvd_result.has_exposure if adcvd_result else False,
        has_fta_preference=fta_preference_pct > 0,
        has_active_exclusion=exclusion_result.has_active_exclusion if exclusion_result else False,
        stop_optimization=stop_opt,
        base_rate_source=base_rate_source,
        base_rate_warning=base_rate_warning,
    )


def compute_duty_delta(
    baseline: DutyBreakdown,
    optimized: DutyBreakdown,
    *,
    declared_value_per_unit: float = 100.0,
    annual_volume: int | None = None,
) -> dict[str, Any]:
    """Compute duty savings between baseline and optimized scenarios.

    Returns a dict with savings breakdown across all duty layers.
    """
    base_delta = baseline.base_rate - optimized.base_rate
    effective_base_delta = baseline.effective_base_rate - optimized.effective_base_rate
    overlay_delta = baseline.overlay_total - optimized.overlay_total
    adcvd_delta = baseline.trade_remedy_total - optimized.trade_remedy_total
    total_delta = baseline.total_duty_rate - optimized.total_duty_rate

    savings_per_unit = total_delta / 100.0 * declared_value_per_unit
    annual_savings = savings_per_unit * annual_volume if annual_volume is not None else None

    return {
        "base_rate_delta": base_delta,
        "effective_base_rate_delta": effective_base_delta,
        "overlay_delta": overlay_delta,
        "adcvd_delta": adcvd_delta,
        "total_rate_delta": total_delta,
        "savings_per_unit": savings_per_unit,
        "annual_savings": annual_savings,
        "baseline_total": baseline.total_duty_rate,
        "optimized_total": optimized.total_duty_rate,
        "fta_savings_pct": optimized.fta_preference_pct - baseline.fta_preference_pct,
        "exclusion_savings_pct": optimized.exclusion_relief_rate - baseline.exclusion_relief_rate,
    }
