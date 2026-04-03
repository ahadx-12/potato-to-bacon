"""Strategy Synthesizer — the brain of the Tariff Engineer.

A tariff calculator returns duty rates.
A tariff engineer returns STRATEGIES: named, cross-portfolio action plans
that group related opportunities, sequence implementation steps, calculate
portfolio-wide ROI, and tell the importer exactly what to do.

This module takes the raw per-SKU opportunities produced by the mutation
engine, FTA engine, AD/CVD registry, and exclusion tracker, and synthesizes
them into named engineering strategies.

Example output:
  Strategy: "USMCA Preference Program"
    Affected SKUs: 14 of 47 (steel brackets, aluminum housings, ...)
    Combined annual savings: $340,000
    Implementation: File USMCA certificates of origin for MX-origin items
    Complexity: LOW (documentation only)
    Timeline: 2-4 weeks
    Prerequisite: Confirm RVC >= 75% for all affected SKUs

  Strategy: "Reclassify Composite Products per GRI 3(b)"
    Affected SKUs: 3 (USB cable assembly, motor controller, LED panel)
    Combined annual savings: $87,000
    Implementation: Request binding ruling, amend future entries
    Complexity: MEDIUM (requires CBP ruling)
    Timeline: 6-12 months
    Prerequisite: Obtain material weight breakdowns from suppliers
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from potatobacon.tariff.engineering_opportunity import (
    OpportunityType,
    TariffEngineeringOpportunity,
)


# ---------------------------------------------------------------------------
# Strategy classification
# ---------------------------------------------------------------------------

class StrategyComplexity(str, Enum):
    """Implementation complexity for a strategy."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StrategyPhase(str, Enum):
    """When this strategy should be executed in an engagement timeline."""

    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


# ---------------------------------------------------------------------------
# Strategy template registry
# ---------------------------------------------------------------------------

_STRATEGY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "usmca_preference": {
        "name": "USMCA Duty Preference Program",
        "opportunity_types": {OpportunityType.FTA_UTILIZATION},
        "fta_keywords": {"usmca", "nafta", "canada", "mexico", "ca", "mx"},
        "complexity": StrategyComplexity.LOW,
        "phase": StrategyPhase.IMMEDIATE,
        "timeline_days": (14, 30),
        "implementation_steps": [
            "Audit all MX/CA-origin SKUs for USMCA eligibility",
            "Collect certificates of origin from each supplier",
            "Verify Regional Value Content (RVC) meets threshold per heading",
            "File USMCA preference claims on future entries",
            "Amend prior entries within protest window (180 days) for retroactive savings",
        ],
        "prerequisites": [
            "Supplier certificates of origin (USMCA format)",
            "RVC calculation worksheets per HTS heading",
            "Proof of direct shipment (bill of lading)",
        ],
        "estimated_cost_per_sku": 500.0,
    },
    "gsp_preference": {
        "name": "GSP Duty Elimination Program",
        "opportunity_types": {OpportunityType.FTA_UTILIZATION},
        "fta_keywords": {"gsp", "generalized system"},
        "complexity": StrategyComplexity.LOW,
        "phase": StrategyPhase.IMMEDIATE,
        "timeline_days": (7, 21),
        "implementation_steps": [
            "Confirm GSP-eligible country of origin for each SKU",
            "Verify product is on GSP-eligible list for that country",
            "Obtain country of origin certificates",
            "Claim GSP preference on entry summary (SPI 'A')",
        ],
        "prerequisites": [
            "Country of origin documentation",
            "GSP eligibility confirmation per HTS code",
        ],
        "estimated_cost_per_sku": 200.0,
    },
    "reclassification_gri": {
        "name": "HTS Reclassification per GRI Rules",
        "opportunity_types": {OpportunityType.RECLASSIFICATION},
        "fta_keywords": set(),
        "complexity": StrategyComplexity.MEDIUM,
        "phase": StrategyPhase.SHORT_TERM,
        "timeline_days": (60, 180),
        "implementation_steps": [
            "Document product composition and function for GRI analysis",
            "Prepare classification argument citing specific GRI rule",
            "Consider requesting CBP binding ruling for certainty",
            "File entries under revised HTS code with supporting documentation",
            "Amend prior entries within protest window if applicable",
        ],
        "prerequisites": [
            "Detailed product specifications and material breakdown",
            "GRI analysis memorandum from licensed customs broker",
            "Optionally: CBP binding ruling (Form 177)",
        ],
        "estimated_cost_per_sku": 3000.0,
    },
    "product_modification": {
        "name": "Product Engineering for Tariff Optimization",
        "opportunity_types": {OpportunityType.PRODUCT_ENGINEERING},
        "fta_keywords": set(),
        "complexity": StrategyComplexity.HIGH,
        "phase": StrategyPhase.MEDIUM_TERM,
        "timeline_days": (90, 365),
        "implementation_steps": [
            "Identify specific product modification that shifts HTS classification",
            "Validate modification with engineering/R&D team for feasibility",
            "Prototype modified product and confirm HTS classification change",
            "Request CBP binding ruling on modified product",
            "Update supplier specs and begin production of modified version",
            "File entries under new HTS code with ruling reference",
        ],
        "prerequisites": [
            "Engineering feasibility assessment",
            "Cost analysis of product modification",
            "CBP binding ruling on modified product",
            "Updated product specifications from supplier",
        ],
        "estimated_cost_per_sku": 15000.0,
    },
    "origin_shift": {
        "name": "Supply Chain Origin Restructuring",
        "opportunity_types": {OpportunityType.TRADE_LANE},
        "fta_keywords": set(),
        "complexity": StrategyComplexity.HIGH,
        "phase": StrategyPhase.LONG_TERM,
        "timeline_days": (180, 730),
        "implementation_steps": [
            "Identify target origin country that eliminates Section 301/AD-CVD exposure",
            "Qualify alternative suppliers in target country",
            "Validate substantial transformation occurs in target country",
            "Establish quality control and logistics for new supply chain",
            "Transition production and begin importing from new origin",
            "Monitor CBP scrutiny of origin claims (country-of-origin audits)",
        ],
        "prerequisites": [
            "Supplier qualification in target country",
            "Substantial transformation analysis",
            "Cost-benefit analysis including logistics changes",
            "Legal review of origin determination rules",
        ],
        "estimated_cost_per_sku": 50000.0,
    },
    "exclusion_capture": {
        "name": "Section 232/301 Exclusion Filing",
        "opportunity_types": {OpportunityType.EXCLUSION_FILING},
        "fta_keywords": set(),
        "complexity": StrategyComplexity.LOW,
        "phase": StrategyPhase.IMMEDIATE,
        "timeline_days": (7, 14),
        "implementation_steps": [
            "Confirm active exclusion covers HTS code + origin combination",
            "File exclusion claim on entry summary",
            "Amend prior entries to capture retroactive relief",
        ],
        "prerequisites": [
            "Verification that exclusion is still active and not expired",
            "Entry documentation matching exclusion scope",
        ],
        "estimated_cost_per_sku": 300.0,
    },
    "documentation_quick_win": {
        "name": "Documentation-Only Duty Reduction",
        "opportunity_types": {OpportunityType.DOCUMENTATION},
        "fta_keywords": set(),
        "complexity": StrategyComplexity.LOW,
        "phase": StrategyPhase.IMMEDIATE,
        "timeline_days": (7, 21),
        "implementation_steps": [
            "Identify missing documentation that supports lower duty classification",
            "Request required certificates/lab reports from suppliers",
            "Submit documentation to customs broker for entry amendment",
            "File amended entries for retroactive savings where applicable",
        ],
        "prerequisites": [
            "Supplier cooperation for documentation requests",
            "Customs broker engagement for entry amendments",
        ],
        "estimated_cost_per_sku": 500.0,
    },
    "adcvd_remediation": {
        "name": "AD/CVD Exposure Remediation",
        "opportunity_types": {OpportunityType.AD_CVD_EXPOSURE, OpportunityType.AD_CVD_ENGINEERING},
        "fta_keywords": set(),
        "complexity": StrategyComplexity.HIGH,
        "phase": StrategyPhase.IMMEDIATE,
        "timeline_days": (30, 90),
        "implementation_steps": [
            "Confirm AD/CVD order scope applicability to affected SKUs",
            "Engage trade remedy counsel for scope determination",
            "If in scope: calculate underpayment and consider prior disclosure",
            "If scope is ambiguous: request scope ruling from Commerce Department",
            "Implement prospective compliance (correct duty deposits)",
            "Evaluate origin shift or product modification to exit scope",
        ],
        "prerequisites": [
            "Trade remedy counsel engagement",
            "Detailed import history for affected HTS codes",
            "Product specifications for scope analysis",
        ],
        "estimated_cost_per_sku": 10000.0,
    },
}


# ---------------------------------------------------------------------------
# Strategy data model
# ---------------------------------------------------------------------------

@dataclass
class StrategyROI:
    """Return on investment calculation for a strategy."""

    total_implementation_cost: float
    annual_savings: float
    payback_months: Optional[float]
    three_year_net_savings: float
    roi_pct: float


@dataclass
class EngineeringStrategy:
    """A named, cross-portfolio tariff engineering strategy.

    This is the primary deliverable of a tariff ENGINEER.
    It groups related opportunities across SKUs into a single
    actionable program with sequenced steps and ROI.
    """

    strategy_id: str
    strategy_key: str
    name: str
    complexity: StrategyComplexity
    phase: StrategyPhase
    timeline_min_days: int
    timeline_max_days: int

    affected_skus: List[str]
    affected_sku_count: int
    total_skus_in_portfolio: int
    coverage_pct: float

    opportunities: List[TariffEngineeringOpportunity]
    opportunity_types: List[str]

    combined_baseline_rate_avg: float
    combined_optimized_rate_avg: float
    combined_rate_reduction_avg: float
    combined_annual_savings: Optional[float]

    implementation_steps: List[str]
    prerequisites: List[str]
    evidence_required: List[str]

    roi: Optional[StrategyROI]

    risk_grade: str
    confidence: str
    requires_professional_review: bool

    executive_summary: str


@dataclass
class StrategyPortfolio:
    """Complete strategy portfolio for an importer.

    This is the top-level "what should we do?" deliverable.
    """

    strategies: List[EngineeringStrategy]
    total_strategies: int
    immediate_actions: List[EngineeringStrategy]
    short_term_actions: List[EngineeringStrategy]
    medium_term_actions: List[EngineeringStrategy]
    long_term_actions: List[EngineeringStrategy]

    combined_annual_savings: Optional[float]
    combined_implementation_cost: Optional[float]
    portfolio_roi: Optional[StrategyROI]

    executive_briefing: str


# ---------------------------------------------------------------------------
# Core synthesis logic
# ---------------------------------------------------------------------------

def _match_strategy_template(
    opp: TariffEngineeringOpportunity,
) -> str:
    """Determine which strategy template an opportunity belongs to."""

    opp_type = opp.opportunity_type
    title_lower = opp.title.lower() if opp.title else ""
    desc_lower = opp.description.lower() if opp.description else ""
    combined_text = f"{title_lower} {desc_lower}"

    for key, template in _STRATEGY_TEMPLATES.items():
        if opp_type not in template["opportunity_types"]:
            continue
        keywords = template["fta_keywords"]
        if not keywords:
            return key
        if any(kw in combined_text for kw in keywords):
            return key

    type_fallback = {
        OpportunityType.RECLASSIFICATION: "reclassification_gri",
        OpportunityType.PRODUCT_ENGINEERING: "product_modification",
        OpportunityType.TRADE_LANE: "origin_shift",
        OpportunityType.AD_CVD_EXPOSURE: "adcvd_remediation",
        OpportunityType.AD_CVD_ENGINEERING: "adcvd_remediation",
        OpportunityType.FTA_UTILIZATION: "gsp_preference",
        OpportunityType.EXCLUSION_FILING: "exclusion_capture",
        OpportunityType.DOCUMENTATION: "documentation_quick_win",
    }
    return type_fallback.get(opp_type, "documentation_quick_win")


def _compute_roi(
    opportunities: Sequence[TariffEngineeringOpportunity],
    cost_per_sku: float,
) -> Optional[StrategyROI]:
    """Compute ROI for a group of opportunities."""

    annual_savings = sum(
        o.annual_savings_estimate for o in opportunities
        if o.annual_savings_estimate is not None
    )
    if annual_savings <= 0:
        return None

    sku_count = len({o.sku_id for o in opportunities if o.sku_id})
    total_cost = cost_per_sku * max(sku_count, 1)

    monthly_savings = annual_savings / 12.0
    payback = total_cost / monthly_savings if monthly_savings > 0 else None
    three_year_net = (annual_savings * 3) - total_cost
    roi_pct = (three_year_net / total_cost * 100) if total_cost > 0 else 0.0

    return StrategyROI(
        total_implementation_cost=round(total_cost, 2),
        annual_savings=round(annual_savings, 2),
        payback_months=round(payback, 1) if payback is not None else None,
        three_year_net_savings=round(three_year_net, 2),
        roi_pct=round(roi_pct, 1),
    )


def _worst_risk(opportunities: Sequence[TariffEngineeringOpportunity]) -> str:
    """Return the worst (highest) risk grade across opportunities."""

    grades = [o.risk_grade.value if hasattr(o.risk_grade, "value") else str(o.risk_grade) for o in opportunities]
    for grade in ("C", "B", "A"):
        if grade in grades:
            return grade
    return "B"


def _lowest_confidence(opportunities: Sequence[TariffEngineeringOpportunity]) -> str:
    """Return the lowest confidence across opportunities."""

    values = [o.confidence.value if hasattr(o.confidence, "value") else str(o.confidence) for o in opportunities]
    for level in ("low", "medium", "high"):
        if level in values:
            return level
    return "medium"


def _build_executive_summary(
    name: str,
    affected_count: int,
    total_count: int,
    savings: Optional[float],
    complexity: StrategyComplexity,
    phase: StrategyPhase,
    timeline_min: int,
    timeline_max: int,
) -> str:
    """Generate a plain-English executive summary for a strategy."""

    savings_str = f"${savings:,.0f}" if savings else "TBD (provide unit values and volumes)"
    pct = round(affected_count / total_count * 100) if total_count > 0 else 0

    phase_label = {
        StrategyPhase.IMMEDIATE: "can begin immediately",
        StrategyPhase.SHORT_TERM: "targets the next 1-3 months",
        StrategyPhase.MEDIUM_TERM: "requires 3-12 months to implement",
        StrategyPhase.LONG_TERM: "is a 12+ month strategic initiative",
    }[phase]

    timeline_str = (
        f"{timeline_min}-{timeline_max} days"
        if timeline_max <= 90
        else f"{timeline_min // 30}-{timeline_max // 30} months"
    )

    return (
        f"{name}: affects {affected_count} of {total_count} SKUs ({pct}% of portfolio). "
        f"Estimated annual savings: {savings_str}. "
        f"Complexity: {complexity.value}. "
        f"This strategy {phase_label} with an estimated timeline of {timeline_str}."
    )


def synthesize_strategies(
    opportunities: Sequence[TariffEngineeringOpportunity],
    total_portfolio_skus: int,
) -> StrategyPortfolio:
    """Synthesize raw opportunities into named engineering strategies.

    This is the core function that transforms a tariff calculator into
    a tariff engineer.  It:
      1. Groups opportunities by strategy template
      2. Computes cross-SKU ROI for each strategy
      3. Sequences strategies by phase (immediate -> long-term)
      4. Generates executive summaries and implementation roadmaps
    """

    grouped: Dict[str, List[TariffEngineeringOpportunity]] = defaultdict(list)
    for opp in opportunities:
        key = _match_strategy_template(opp)
        grouped[key].append(opp)

    strategies: List[EngineeringStrategy] = []

    for strategy_key, opps in grouped.items():
        template = _STRATEGY_TEMPLATES.get(strategy_key)
        if not template:
            continue

        affected_skus = sorted({o.sku_id for o in opps if o.sku_id})
        affected_count = len(affected_skus) or len(opps)

        rates = [o.baseline_total_rate for o in opps if o.baseline_total_rate > 0]
        opt_rates = [o.optimized_total_rate for o in opps]
        reductions = [o.rate_reduction_pct for o in opps if o.rate_reduction_pct > 0]

        avg_baseline = sum(rates) / len(rates) if rates else 0.0
        avg_optimized = sum(opt_rates) / len(opt_rates) if opt_rates else 0.0
        avg_reduction = sum(reductions) / len(reductions) if reductions else 0.0

        annual_savings = sum(
            o.annual_savings_estimate for o in opps
            if o.annual_savings_estimate is not None
        )
        annual_savings_or_none = annual_savings if annual_savings > 0 else None

        all_evidence: List[str] = []
        seen_evidence: set[str] = set()
        for opp in opps:
            for ev in opp.evidence_required:
                if ev not in seen_evidence:
                    all_evidence.append(ev)
                    seen_evidence.add(ev)

        complexity = template["complexity"]
        phase = template["phase"]
        timeline_min, timeline_max = template["timeline_days"]
        cost_per_sku = template["estimated_cost_per_sku"]

        roi = _compute_roi(opps, cost_per_sku)

        opp_types = sorted({
            o.opportunity_type.value
            if hasattr(o.opportunity_type, "value")
            else str(o.opportunity_type)
            for o in opps
        })

        coverage = round(affected_count / total_portfolio_skus * 100, 1) if total_portfolio_skus > 0 else 0.0

        summary = _build_executive_summary(
            name=template["name"],
            affected_count=affected_count,
            total_count=total_portfolio_skus,
            savings=annual_savings_or_none,
            complexity=complexity,
            phase=phase,
            timeline_min=timeline_min,
            timeline_max=timeline_max,
        )

        strategy = EngineeringStrategy(
            strategy_id=str(uuid.uuid4()),
            strategy_key=strategy_key,
            name=template["name"],
            complexity=complexity,
            phase=phase,
            timeline_min_days=timeline_min,
            timeline_max_days=timeline_max,
            affected_skus=affected_skus,
            affected_sku_count=affected_count,
            total_skus_in_portfolio=total_portfolio_skus,
            coverage_pct=coverage,
            opportunities=list(opps),
            opportunity_types=opp_types,
            combined_baseline_rate_avg=round(avg_baseline, 4),
            combined_optimized_rate_avg=round(avg_optimized, 4),
            combined_rate_reduction_avg=round(avg_reduction, 4),
            combined_annual_savings=annual_savings_or_none,
            implementation_steps=list(template["implementation_steps"]),
            prerequisites=list(template["prerequisites"]),
            evidence_required=all_evidence,
            roi=roi,
            risk_grade=_worst_risk(opps),
            confidence=_lowest_confidence(opps),
            requires_professional_review=any(o.requires_professional_review for o in opps),
            executive_summary=summary,
        )
        strategies.append(strategy)

    _PHASE_ORDER = {
        StrategyPhase.IMMEDIATE: 0,
        StrategyPhase.SHORT_TERM: 1,
        StrategyPhase.MEDIUM_TERM: 2,
        StrategyPhase.LONG_TERM: 3,
    }
    strategies.sort(key=lambda s: (
        _PHASE_ORDER.get(s.phase, 99),
        -(s.combined_annual_savings or 0),
    ))

    immediate = [s for s in strategies if s.phase == StrategyPhase.IMMEDIATE]
    short_term = [s for s in strategies if s.phase == StrategyPhase.SHORT_TERM]
    medium_term = [s for s in strategies if s.phase == StrategyPhase.MEDIUM_TERM]
    long_term = [s for s in strategies if s.phase == StrategyPhase.LONG_TERM]

    total_savings = sum(s.combined_annual_savings or 0 for s in strategies)
    total_cost = sum(
        s.roi.total_implementation_cost if s.roi else 0
        for s in strategies
    )

    portfolio_roi: Optional[StrategyROI] = None
    if total_savings > 0:
        monthly = total_savings / 12.0
        payback = total_cost / monthly if monthly > 0 else None
        three_year = (total_savings * 3) - total_cost
        roi_pct = (three_year / total_cost * 100) if total_cost > 0 else 0.0
        portfolio_roi = StrategyROI(
            total_implementation_cost=round(total_cost, 2),
            annual_savings=round(total_savings, 2),
            payback_months=round(payback, 1) if payback is not None else None,
            three_year_net_savings=round(three_year, 2),
            roi_pct=round(roi_pct, 1),
        )

    immediate_count = len(immediate)
    immediate_savings = sum(s.combined_annual_savings or 0 for s in immediate)
    briefing_parts = [
        f"Identified {len(strategies)} engineering strategies across your portfolio.",
    ]
    if immediate_count > 0:
        briefing_parts.append(
            f"{immediate_count} strategies can begin immediately"
            + (f", capturing ${immediate_savings:,.0f}/year." if immediate_savings > 0 else ".")
        )
    if total_savings > 0:
        briefing_parts.append(
            f"Total achievable annual savings: ${total_savings:,.0f}."
        )
    if portfolio_roi and portfolio_roi.payback_months is not None:
        briefing_parts.append(
            f"Portfolio payback period: {portfolio_roi.payback_months:.0f} months."
        )
    briefing = " ".join(briefing_parts)

    return StrategyPortfolio(
        strategies=strategies,
        total_strategies=len(strategies),
        immediate_actions=immediate,
        short_term_actions=short_term,
        medium_term_actions=medium_term,
        long_term_actions=long_term,
        combined_annual_savings=total_savings if total_savings > 0 else None,
        combined_implementation_cost=total_cost if total_cost > 0 else None,
        portfolio_roi=portfolio_roi,
        executive_briefing=briefing,
    )
