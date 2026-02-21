"""Portfolio-level BOM engineering report.

This module produces the primary deliverable of a tariff engineering engagement:
a ranked list of savings opportunities across an importer's full product portfolio,
with a clear executive summary of total exposure and achievable savings.

A tariff ENGINEER's deliverable is not "here are your duty rates."
It is:
  "Your annual duty burden is $4.2M.
   We identified $1.1M in achievable savings.
   Here are your top 5 opportunities, ranked by value.
   Three of them require only documentation — you can start today."

This module builds that report from per-SKU analysis results produced by
the TEaaS pipeline.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from potatobacon.tariff.engineering_opportunity import (
    OpportunityConfidence,
    OpportunityType,
    SKUDutyExposure,
    TariffEngineeringOpportunity,
    build_opportunity_from_adcvd_exposure,
    build_opportunity_from_exclusion,
    build_opportunity_from_fta,
    build_opportunity_from_mutation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report models (Pydantic-free dataclasses for lightweight use)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSummary:
    """High-level numbers for the entire BOM."""

    total_skus: int
    skus_analyzed: int
    skus_with_opportunities: int
    skus_with_adcvd_exposure: int

    # Duty burden (requires declared_value and annual_volume per SKU)
    total_annual_duty_exposure: Optional[float]   # USD; None if no value/volume data
    achievable_annual_savings: Optional[float]    # USD; sum of top opportunity per SKU

    # Rate summary
    weighted_avg_baseline_rate: Optional[float]   # % duty; value-weighted
    weighted_avg_optimized_rate: Optional[float]  # % duty after best opportunity per SKU

    # Opportunity counts
    documentation_only_count: int    # Quick wins requiring no physical changes
    product_engineering_count: int
    trade_lane_count: int
    reclassification_count: int
    fta_utilization_count: int
    exclusion_filing_count: int
    adcvd_exposure_count: int


@dataclass
class BOMEngineeringReport:
    """Full tariff engineering report for a BOM portfolio.

    This is the top-level output of a tariff engineering engagement.
    It contains:
      - An executive summary (portfolio_summary)
      - All findings sorted by annual_savings_estimate descending
      - Quick-win findings (documentation-only, can act immediately)
      - Per-SKU detailed findings
    """

    # Identity
    report_id: str
    law_context: str
    tariff_manifest_hash: str
    tenant_id: Optional[str]
    analyzed_at: str

    # Summary
    portfolio_summary: PortfolioSummary

    # All opportunities, sorted: risk findings last, then by savings descending
    all_opportunities: List[TariffEngineeringOpportunity]

    # Quick wins: documentation-only, HIGH confidence, no physical changes
    quick_wins: List[TariffEngineeringOpportunity]

    # Per-SKU detailed view
    sku_findings: List[SKUDutyExposure]

    # Warnings / coverage notes from the engine
    warnings: List[str]


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_bom_engineering_report(
    *,
    report_id: str,
    law_context: str,
    tariff_manifest_hash: str,
    tenant_id: Optional[str],
    analyzed_at: str,
    sku_results: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> BOMEngineeringReport:
    """Build a BOMEngineeringReport from per-SKU analysis results.

    Parameters
    ----------
    sku_results
        List of dicts, each containing:
          - sku_id: Optional[str]
          - description: str
          - origin_country: Optional[str]
          - current_hts_code: Optional[str]
          - inferred_category: Optional[str]
          - declared_value_per_unit: Optional[float]
          - annual_volume: Optional[int]
          - baseline_total_rate: float
          - optimized_total_rate: float
          - has_adcvd_exposure: bool
          - has_fta_preference: bool
          - section_232_rate: float
          - section_301_rate: float
          - ad_duty_rate: float
          - cvd_duty_rate: float
          - exclusion_relief_rate: float
          - fta_preference_pct: float
          - mutation_results: List[Dict]   # from TEaaS pipeline
          - adcvd_orders: List[Dict]       # matched orders if any
          - fta_result: Optional[Dict]     # FTA eligibility if any
          - exclusion_result: Optional[Dict]
    """
    all_opportunities: List[TariffEngineeringOpportunity] = []
    sku_findings: List[SKUDutyExposure] = []
    opp_counter = [0]

    def _next_id(sku_id: Optional[str]) -> str:
        opp_counter[0] += 1
        prefix = sku_id[:8] if sku_id else "SKU"
        return f"{prefix}-OPP-{opp_counter[0]:04d}"

    total_annual_duty = 0.0
    has_any_value_data = False
    weighted_rate_num = 0.0
    weighted_rate_den = 0.0
    weighted_opt_num = 0.0

    for result in sku_results:
        sku_id = result.get("sku_id")
        description = result.get("description", "Unknown product")
        origin = result.get("origin_country")
        hts_code = result.get("current_hts_code") or ""
        inferred_cat = result.get("inferred_category", "other")
        declared_value = result.get("declared_value_per_unit")
        volume = result.get("annual_volume")

        baseline_rate = float(result.get("baseline_total_rate", 0.0))
        optimized_rate = float(result.get("optimized_total_rate", baseline_rate))
        section_232 = float(result.get("section_232_rate", 0.0))
        section_301 = float(result.get("section_301_rate", 0.0))
        ad_rate = float(result.get("ad_duty_rate", 0.0))
        cvd_rate = float(result.get("cvd_duty_rate", 0.0))
        exclusion_relief = float(result.get("exclusion_relief_rate", 0.0))
        fta_pct = float(result.get("fta_preference_pct", 0.0))

        # Annual exposure calculation
        annual_duty = None
        if declared_value is not None and volume is not None:
            annual_duty = declared_value * volume * baseline_rate / 100.0
            total_annual_duty += annual_duty
            has_any_value_data = True
            weight = declared_value * volume
            weighted_rate_num += baseline_rate * weight
            weighted_opt_num += optimized_rate * weight
            weighted_rate_den += weight

        sku_opps: List[TariffEngineeringOpportunity] = []

        # --- Build opportunities from mutation results ---
        for mut in result.get("mutation_results", []):
            mut_rate = float(mut.get("effective_duty_rate", mut.get("projected_duty_rate", baseline_rate)))
            effective_savings = float(mut.get("effective_savings", 0.0))
            if effective_savings <= 0:
                continue
            opp = build_opportunity_from_mutation(
                opportunity_id=_next_id(sku_id),
                sku_id=sku_id,
                mutation_description=str(mut.get("human_description", "Optimization candidate")),
                fact_patch=dict(mut.get("fact_patch", {})),
                baseline_total_rate=baseline_rate,
                optimized_total_rate=mut_rate,
                declared_value_per_unit=declared_value,
                annual_volume=volume,
                verified=bool(mut.get("verified", True)),
            )
            sku_opps.append(opp)

        # --- FTA utilization opportunity ---
        fta_result = result.get("fta_result")
        if fta_result and fta_result.get("has_eligible_program"):
            best = fta_result.get("best_program") or {}
            program_id = best.get("program_id", "FTA")
            program_name = best.get("program_name", program_id)
            pref_pct = float(best.get("preference_pct", fta_result.get("best_preference_pct", 100.0)))
            base_rate_for_fta = float(result.get("base_rate", baseline_rate))
            if pref_pct > 0 and base_rate_for_fta > 0 and fta_pct < pref_pct:
                missing_reqs = list(best.get("missing_requirements", []))
                opp = build_opportunity_from_fta(
                    opportunity_id=_next_id(sku_id),
                    sku_id=sku_id,
                    description=description,
                    hts_code=hts_code,
                    origin_country=origin or "",
                    import_country=result.get("import_country", "US"),
                    program_id=program_id,
                    program_name=program_name,
                    preference_pct=pref_pct,
                    base_rate=base_rate_for_fta,
                    current_total_rate=baseline_rate,
                    declared_value_per_unit=declared_value,
                    annual_volume=volume,
                    missing_requirements=missing_reqs,
                )
                sku_opps.append(opp)

        # --- AD/CVD exposure ---
        adcvd_orders = result.get("adcvd_orders", [])
        if adcvd_orders or (ad_rate > 0 or cvd_rate > 0):
            order_ids = [o.get("order_id", "?") for o in adcvd_orders] if adcvd_orders else []
            opp = build_opportunity_from_adcvd_exposure(
                opportunity_id=_next_id(sku_id),
                sku_id=sku_id,
                description=description,
                hts_code=hts_code,
                origin_country=origin or "",
                ad_rate=ad_rate,
                cvd_rate=cvd_rate,
                order_ids=order_ids,
                confidence=result.get("adcvd_confidence", "medium"),
                current_total_rate=baseline_rate - ad_rate - cvd_rate,
                declared_value_per_unit=declared_value,
                annual_volume=volume,
            )
            sku_opps.append(opp)

        # --- Exclusion ---
        excl_result = result.get("exclusion_result")
        if excl_result and excl_result.get("has_active_exclusion"):
            relief = float(excl_result.get("total_exclusion_relief_pct", 0.0))
            excl_id = excl_result.get("exclusion_id", "exclusion")
            if relief > 0:
                opp = build_opportunity_from_exclusion(
                    opportunity_id=_next_id(sku_id),
                    sku_id=sku_id,
                    description=description,
                    hts_code=hts_code,
                    origin_country=origin,
                    exclusion_id=excl_id,
                    relief_pct=relief,
                    current_total_rate=baseline_rate,
                    declared_value_per_unit=declared_value,
                    annual_volume=volume,
                )
                sku_opps.append(opp)

        # Sort per-SKU opportunities: risk findings last, then by savings descending
        sku_opps = _sort_opportunities(sku_opps)

        # Best achievable for this SKU
        savings_opps = [o for o in sku_opps if not o.is_risk_finding]
        best_savings_pct = max((o.rate_reduction_pct for o in savings_opps), default=0.0)
        best_opt_rate = baseline_rate - best_savings_pct

        sku_exposure = SKUDutyExposure(
            sku_id=sku_id,
            description=description,
            origin_country=origin,
            current_hts_code=hts_code or None,
            inferred_category=inferred_cat,
            base_rate=float(result.get("base_rate", 0.0)),
            section_232_rate=section_232,
            section_301_rate=section_301,
            ad_duty_rate=ad_rate,
            cvd_duty_rate=cvd_rate,
            exclusion_relief_rate=exclusion_relief,
            fta_preference_pct=fta_pct,
            total_effective_rate=baseline_rate,
            has_trade_remedy_exposure=(section_232 > 0 or section_301 > 0),
            has_adcvd_exposure=(ad_rate > 0 or cvd_rate > 0),
            has_fta_opportunity=(fta_pct == 0.0 and bool(
                fta_result and fta_result.get("has_eligible_program")
            )),
            requires_manual_review=bool(result.get("requires_manual_review", False)),
            opportunities=sku_opps,
            best_savings_pct=round(best_savings_pct, 4),
        )
        sku_findings.append(sku_exposure)
        all_opportunities.extend(sku_opps)

    # Global sort
    all_opportunities = _sort_opportunities(all_opportunities)

    # Quick wins: documentation-only, high/medium confidence, no physical changes
    quick_wins = [
        o for o in all_opportunities
        if not o.is_risk_finding
        and o.opportunity_type in (OpportunityType.DOCUMENTATION, OpportunityType.FTA_UTILIZATION, OpportunityType.EXCLUSION_FILING)
        and o.confidence in (OpportunityConfidence.HIGH, OpportunityConfidence.MEDIUM)
    ]

    # Achievable savings: best opportunity per SKU
    achievable_savings = _compute_achievable_savings(sku_findings)

    # Weighted average rates
    weighted_avg_baseline = (
        round(weighted_rate_num / weighted_rate_den, 4) if weighted_rate_den > 0 else None
    )
    weighted_avg_optimized = (
        round(weighted_opt_num / weighted_rate_den, 4) if weighted_rate_den > 0 else None
    )

    summary = PortfolioSummary(
        total_skus=len(sku_results),
        skus_analyzed=len(sku_results),
        skus_with_opportunities=sum(
            1 for s in sku_findings
            if any(not o.is_risk_finding for o in s.opportunities)
        ),
        skus_with_adcvd_exposure=sum(1 for s in sku_findings if s.has_adcvd_exposure),
        total_annual_duty_exposure=round(total_annual_duty, 2) if has_any_value_data else None,
        achievable_annual_savings=round(achievable_savings, 2) if has_any_value_data else None,
        weighted_avg_baseline_rate=weighted_avg_baseline,
        weighted_avg_optimized_rate=weighted_avg_optimized,
        documentation_only_count=sum(
            1 for o in all_opportunities
            if o.opportunity_type == OpportunityType.DOCUMENTATION
        ),
        product_engineering_count=sum(
            1 for o in all_opportunities
            if o.opportunity_type == OpportunityType.PRODUCT_ENGINEERING
        ),
        trade_lane_count=sum(
            1 for o in all_opportunities
            if o.opportunity_type == OpportunityType.TRADE_LANE
        ),
        reclassification_count=sum(
            1 for o in all_opportunities
            if o.opportunity_type == OpportunityType.RECLASSIFICATION
        ),
        fta_utilization_count=sum(
            1 for o in all_opportunities
            if o.opportunity_type == OpportunityType.FTA_UTILIZATION
        ),
        exclusion_filing_count=sum(
            1 for o in all_opportunities
            if o.opportunity_type == OpportunityType.EXCLUSION_FILING
        ),
        adcvd_exposure_count=sum(
            1 for o in all_opportunities
            if o.opportunity_type == OpportunityType.AD_CVD_EXPOSURE
        ),
    )

    return BOMEngineeringReport(
        report_id=report_id,
        law_context=law_context,
        tariff_manifest_hash=tariff_manifest_hash,
        tenant_id=tenant_id,
        analyzed_at=analyzed_at,
        portfolio_summary=summary,
        all_opportunities=all_opportunities,
        quick_wins=quick_wins,
        sku_findings=sku_findings,
        warnings=warnings or [],
    )


def _sort_opportunities(
    opps: List[TariffEngineeringOpportunity],
) -> List[TariffEngineeringOpportunity]:
    """Sort opportunities: savings first (by annual value), risk findings last."""

    def sort_key(o: TariffEngineeringOpportunity) -> Tuple:
        # Risk findings go to the end
        is_risk = 1 if o.is_risk_finding else 0
        # Then sort by annual savings descending (negate for ascending sort)
        annual = -(o.annual_savings_estimate or 0.0)
        # Then by rate reduction descending
        rate = -o.rate_reduction_pct
        # Then by confidence (high first)
        conf_order = {"high": 0, "medium": 1, "low": 2}.get(o.confidence.value, 3)
        return (is_risk, annual, rate, conf_order)

    return sorted(opps, key=sort_key)


def _compute_achievable_savings(sku_findings: List[SKUDutyExposure]) -> float:
    """Sum of the best annual savings opportunity per SKU."""
    total = 0.0
    for sku in sku_findings:
        savings_opps = [
            o for o in sku.opportunities
            if not o.is_risk_finding and o.annual_savings_estimate is not None
        ]
        if savings_opps:
            best = max(o.annual_savings_estimate for o in savings_opps if o.annual_savings_estimate)
            total += best
    return total
