"""Scenario Comparator — side-by-side what-if analysis for tariff engineering.

A tariff calculator gives you one number.
A tariff engineer shows you a MENU of futures:

  | Scenario                    | Duty Rate | Annual Cost | Savings   |
  |-----------------------------|-----------|-------------|-----------|
  | Current State               | 31.5%     | $1,260,000  |     --    |
  | Reclassify per GRI 3(b)     | 6.5%      |   $260,000  | $1,000,000|
  | Shift assembly to MX (USMCA)| 0.0%      |         $0  | $1,260,000|
  | File 301 exclusion          | 6.5%      |   $260,000  | $1,000,000|

This module builds that comparison table from the baseline analysis
and the discovered opportunities.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence

from potatobacon.tariff.engineering_opportunity import (
    OpportunityType,
    TariffEngineeringOpportunity,
)


# ---------------------------------------------------------------------------
# Scenario data model
# ---------------------------------------------------------------------------

@dataclass
class DutyBreakdown:
    """Full duty rate breakdown for a scenario."""

    base_rate: float = 0.0
    section_232_rate: float = 0.0
    section_301_rate: float = 0.0
    ad_duty_rate: float = 0.0
    cvd_duty_rate: float = 0.0
    fta_preference_pct: float = 0.0
    exclusion_relief_rate: float = 0.0
    total_effective_rate: float = 0.0


@dataclass
class ScenarioResult:
    """A single what-if scenario for a SKU or portfolio."""

    scenario_id: str
    scenario_name: str
    scenario_type: str
    description: str
    is_baseline: bool

    duty_breakdown: DutyBreakdown

    annual_duty_cost: Optional[float]
    savings_vs_baseline: Optional[float]
    savings_pct_vs_baseline: Optional[float]

    hts_code: Optional[str]
    origin_country: Optional[str]

    implementation_complexity: str
    implementation_steps: List[str]
    prerequisites: List[str]
    timeline_days: Optional[int]

    source_opportunity: Optional[TariffEngineeringOpportunity]


@dataclass
class SKUScenarioComparison:
    """Side-by-side scenario comparison for a single SKU."""

    sku_id: Optional[str]
    description: str
    declared_value_per_unit: Optional[float]
    annual_volume: Optional[int]

    baseline: ScenarioResult
    alternatives: List[ScenarioResult]
    best_alternative: Optional[ScenarioResult]

    max_savings_pct: float
    max_annual_savings: Optional[float]

    recommendation: str


@dataclass
class PortfolioScenarioComparison:
    """Aggregated scenario comparison across a portfolio."""

    sku_comparisons: List[SKUScenarioComparison]
    total_skus: int
    skus_with_alternatives: int

    baseline_total_annual_cost: Optional[float]
    best_case_total_annual_cost: Optional[float]
    max_portfolio_savings: Optional[float]

    summary: str


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

_TYPE_TO_SCENARIO_NAME = {
    OpportunityType.RECLASSIFICATION: "Reclassify under {target_hts}",
    OpportunityType.PRODUCT_ENGINEERING: "Modify product for {target_hts}",
    OpportunityType.TRADE_LANE: "Shift origin to {target_origin}",
    OpportunityType.FTA_UTILIZATION: "Claim FTA preference",
    OpportunityType.EXCLUSION_FILING: "File Section 232/301 exclusion",
    OpportunityType.DOCUMENTATION: "Provide supporting documentation",
    OpportunityType.AD_CVD_EXPOSURE: "AD/CVD exposure (risk)",
    OpportunityType.AD_CVD_ENGINEERING: "Engineer out of AD/CVD scope",
}

_TYPE_TO_COMPLEXITY = {
    OpportunityType.DOCUMENTATION: "low",
    OpportunityType.FTA_UTILIZATION: "low",
    OpportunityType.EXCLUSION_FILING: "low",
    OpportunityType.RECLASSIFICATION: "medium",
    OpportunityType.PRODUCT_ENGINEERING: "high",
    OpportunityType.TRADE_LANE: "high",
    OpportunityType.AD_CVD_EXPOSURE: "high",
    OpportunityType.AD_CVD_ENGINEERING: "high",
}


def _scenario_name(opp: TariffEngineeringOpportunity) -> str:
    """Generate a human-readable scenario name from an opportunity."""

    template = _TYPE_TO_SCENARIO_NAME.get(
        opp.opportunity_type, "Alternative scenario"
    )
    return template.format(
        target_hts=opp.target_hts_code or "lower-rate heading",
        target_origin=opp.target_origin or opp.current_origin or "alternative country",
    )


def _build_scenario_from_opportunity(
    opp: TariffEngineeringOpportunity,
    baseline_breakdown: DutyBreakdown,
    declared_value: Optional[float],
    annual_volume: Optional[int],
) -> ScenarioResult:
    """Build a ScenarioResult from an opportunity and baseline."""

    rate_reduction = opp.rate_reduction_pct
    opt_rate = opp.optimized_total_rate

    alt_breakdown = DutyBreakdown(
        base_rate=baseline_breakdown.base_rate,
        section_232_rate=baseline_breakdown.section_232_rate,
        section_301_rate=baseline_breakdown.section_301_rate,
        ad_duty_rate=baseline_breakdown.ad_duty_rate,
        cvd_duty_rate=baseline_breakdown.cvd_duty_rate,
        fta_preference_pct=baseline_breakdown.fta_preference_pct,
        exclusion_relief_rate=baseline_breakdown.exclusion_relief_rate,
        total_effective_rate=opt_rate,
    )

    opp_type = opp.opportunity_type
    if opp_type == OpportunityType.RECLASSIFICATION:
        alt_breakdown.base_rate = max(0, baseline_breakdown.base_rate - rate_reduction)
    elif opp_type == OpportunityType.FTA_UTILIZATION:
        alt_breakdown.fta_preference_pct = baseline_breakdown.fta_preference_pct + rate_reduction
    elif opp_type == OpportunityType.EXCLUSION_FILING:
        alt_breakdown.exclusion_relief_rate = baseline_breakdown.exclusion_relief_rate + rate_reduction
    elif opp_type in (OpportunityType.TRADE_LANE, OpportunityType.AD_CVD_ENGINEERING):
        alt_breakdown.section_301_rate = max(0, baseline_breakdown.section_301_rate - rate_reduction)
        alt_breakdown.ad_duty_rate = 0.0
        alt_breakdown.cvd_duty_rate = 0.0

    annual_cost: Optional[float] = None
    savings: Optional[float] = None
    savings_pct: Optional[float] = None

    if declared_value is not None and annual_volume is not None:
        annual_value = declared_value * annual_volume
        annual_cost = round(annual_value * (opt_rate / 100.0), 2)
        baseline_cost = annual_value * (baseline_breakdown.total_effective_rate / 100.0)
        savings = round(baseline_cost - annual_cost, 2)

    if baseline_breakdown.total_effective_rate > 0:
        savings_pct = round(rate_reduction / baseline_breakdown.total_effective_rate * 100, 1)

    return ScenarioResult(
        scenario_id=str(uuid.uuid4()),
        scenario_name=_scenario_name(opp),
        scenario_type=opp.opportunity_type.value if hasattr(opp.opportunity_type, "value") else str(opp.opportunity_type),
        description=opp.description,
        is_baseline=False,
        duty_breakdown=alt_breakdown,
        annual_duty_cost=annual_cost,
        savings_vs_baseline=savings,
        savings_pct_vs_baseline=savings_pct,
        hts_code=opp.target_hts_code or opp.current_hts_code,
        origin_country=opp.target_origin or opp.current_origin,
        implementation_complexity=_TYPE_TO_COMPLEXITY.get(opp.opportunity_type, "medium"),
        implementation_steps=list(opp.action_items),
        prerequisites=list(opp.evidence_required),
        timeline_days=opp.implementation_time_days,
        source_opportunity=opp,
    )


def compare_sku_scenarios(
    *,
    sku_id: Optional[str],
    description: str,
    declared_value_per_unit: Optional[float],
    annual_volume: Optional[int],
    baseline_breakdown: DutyBreakdown,
    baseline_hts_code: Optional[str],
    baseline_origin: Optional[str],
    opportunities: Sequence[TariffEngineeringOpportunity],
) -> SKUScenarioComparison:
    """Build a side-by-side scenario comparison for a single SKU."""

    baseline_cost: Optional[float] = None
    if declared_value_per_unit is not None and annual_volume is not None:
        annual_value = declared_value_per_unit * annual_volume
        baseline_cost = round(annual_value * (baseline_breakdown.total_effective_rate / 100.0), 2)

    baseline = ScenarioResult(
        scenario_id=str(uuid.uuid4()),
        scenario_name="Current State (Baseline)",
        scenario_type="baseline",
        description="Current classification and supply chain configuration",
        is_baseline=True,
        duty_breakdown=baseline_breakdown,
        annual_duty_cost=baseline_cost,
        savings_vs_baseline=None,
        savings_pct_vs_baseline=None,
        hts_code=baseline_hts_code,
        origin_country=baseline_origin,
        implementation_complexity="none",
        implementation_steps=[],
        prerequisites=[],
        timeline_days=None,
        source_opportunity=None,
    )

    alternatives: List[ScenarioResult] = []
    for opp in opportunities:
        if opp.is_risk_finding:
            continue
        if opp.rate_reduction_pct <= 0:
            continue
        alt = _build_scenario_from_opportunity(
            opp, baseline_breakdown, declared_value_per_unit, annual_volume,
        )
        alternatives.append(alt)

    alternatives.sort(key=lambda s: -(s.savings_vs_baseline or 0))

    best = alternatives[0] if alternatives else None
    max_savings_pct = best.savings_pct_vs_baseline or 0.0 if best else 0.0
    max_annual = best.savings_vs_baseline if best else None

    if best and max_annual and max_annual > 0:
        recommendation = (
            f"Best option: {best.scenario_name} — "
            f"saves ${max_annual:,.0f}/year "
            f"({max_savings_pct:.0f}% reduction). "
            f"Complexity: {best.implementation_complexity}."
        )
    elif best:
        recommendation = (
            f"Best option: {best.scenario_name} — "
            f"reduces duty by {best.duty_breakdown.total_effective_rate - baseline_breakdown.total_effective_rate:.1f} "
            f"percentage points. Complexity: {best.implementation_complexity}."
        )
    else:
        recommendation = "No optimization scenarios identified for this SKU."

    return SKUScenarioComparison(
        sku_id=sku_id,
        description=description,
        declared_value_per_unit=declared_value_per_unit,
        annual_volume=annual_volume,
        baseline=baseline,
        alternatives=alternatives,
        best_alternative=best,
        max_savings_pct=max_savings_pct,
        max_annual_savings=max_annual,
        recommendation=recommendation,
    )


def compare_portfolio_scenarios(
    sku_comparisons: Sequence[SKUScenarioComparison],
) -> PortfolioScenarioComparison:
    """Aggregate SKU-level scenario comparisons into a portfolio view."""

    comparisons = list(sku_comparisons)
    total = len(comparisons)
    with_alts = sum(1 for c in comparisons if c.alternatives)

    baseline_total: Optional[float] = None
    best_total: Optional[float] = None

    baseline_costs = [c.baseline.annual_duty_cost for c in comparisons if c.baseline.annual_duty_cost is not None]
    if baseline_costs:
        baseline_total = round(sum(baseline_costs), 2)

    best_costs: List[float] = []
    for c in comparisons:
        if c.best_alternative and c.best_alternative.annual_duty_cost is not None:
            best_costs.append(c.best_alternative.annual_duty_cost)
        elif c.baseline.annual_duty_cost is not None:
            best_costs.append(c.baseline.annual_duty_cost)
    if best_costs:
        best_total = round(sum(best_costs), 2)

    max_savings: Optional[float] = None
    if baseline_total is not None and best_total is not None:
        max_savings = round(baseline_total - best_total, 2)

    parts: List[str] = [f"Analyzed {total} SKUs."]
    if with_alts > 0:
        parts.append(f"{with_alts} have optimization scenarios.")
    if baseline_total is not None:
        parts.append(f"Current annual duty: ${baseline_total:,.0f}.")
    if max_savings is not None and max_savings > 0:
        parts.append(f"Maximum achievable savings: ${max_savings:,.0f}/year.")
    if baseline_total and max_savings and baseline_total > 0:
        pct = max_savings / baseline_total * 100
        parts.append(f"Potential reduction: {pct:.0f}%.")
    summary = " ".join(parts)

    return PortfolioScenarioComparison(
        sku_comparisons=comparisons,
        total_skus=total,
        skus_with_alternatives=with_alts,
        baseline_total_annual_cost=baseline_total,
        best_case_total_annual_cost=best_total,
        max_portfolio_savings=max_savings,
        summary=summary,
    )
