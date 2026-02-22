"""Portfolio-level compliance risk scoring for tariff engineering.

A tariff engineer's job is not only to find savings.  It is equally important
to find compliance RISKS — exposures where the company is underpaying duties
and is therefore subject to CBP audit, assessment, interest, and penalties.

CBP penalty structure:
  - Negligent violation:   2× the unpaid duty
  - Grossly negligent:     4× the unpaid duty
  - Fraudulent violation:  full value of merchandise

Additionally, back-assessed duties run with interest from the date of entry.
A $500K annual AD/CVD exposure at 10% interest for 3 years is $665K before
penalties.

This module scores each SKU — and the portfolio as a whole — on compliance
risk, and produces specific risk findings for inclusion in engineering reports.

Risk categories:
  AD_CVD_UNDERPAYMENT      : AD/CVD exposure confirmed but not being paid
  SECTION_232_UNDERPAYMENT : 232 tariff may apply but not in HTS declaration
  SECTION_301_UNDERPAYMENT : 301 tariff may apply but not in HTS declaration
  ORIGIN_MISREPRESENTATION : Product declared from non-China origin but
                              characteristics suggest possible China origin
  AUDIT_TRIGGER            : Duty payment profile likely to flag for CBP review
  CLASSIFICATION_RISK      : HTS code appears incorrect per GRI analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Risk category and severity
# ---------------------------------------------------------------------------

class RiskCategory(str, Enum):
    AD_CVD_UNDERPAYMENT       = "ad_cvd_underpayment"
    SECTION_232_UNDERPAYMENT  = "section_232_underpayment"
    SECTION_301_UNDERPAYMENT  = "section_301_underpayment"
    ORIGIN_MISREPRESENTATION  = "origin_misrepresentation"
    AUDIT_TRIGGER             = "audit_trigger"
    CLASSIFICATION_RISK       = "classification_risk"
    FTA_OVERCLAIM             = "fta_overclaim"
    VALUATION_RISK            = "valuation_risk"


class RiskSeverity(str, Enum):
    """How urgently this risk should be addressed."""

    CRITICAL   = "critical"   # Immediate action required — active exposure, likely audit
    HIGH       = "high"       # Address within 30 days — significant penalty exposure
    MEDIUM     = "medium"     # Address within 90 days — material but not imminent
    LOW        = "low"        # Monitor — potential exposure but limited immediate risk


# ---------------------------------------------------------------------------
# Risk finding model
# ---------------------------------------------------------------------------

@dataclass
class PortfolioRiskFinding:
    """A single compliance risk finding across the portfolio."""

    risk_id: str
    sku_id: Optional[str]
    description: str

    category: RiskCategory
    severity: RiskSeverity

    # Exposure quantification
    estimated_exposure_pct: float          # Additional duty rate exposure
    estimated_annual_exposure_usd: Optional[float]  # USD if value/volume known

    # Penalty modeling
    potential_penalty_usd: Optional[float]   # Conservative penalty estimate
    penalty_basis: str                        # How penalty was estimated

    # What happened
    risk_summary: str                        # One-line risk statement
    risk_detail: str                         # Full explanation

    # What to do
    immediate_actions: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    legal_citations: List[str] = field(default_factory=list)

    # Metadata
    confidence: str = "medium"              # "high", "medium", "low"
    requires_legal_counsel: bool = False
    prior_disclosure_recommended: bool = False


@dataclass
class PortfolioRiskSummary:
    """Aggregated risk profile for the entire BOM."""

    total_risk_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int

    total_estimated_exposure_usd: Optional[float]
    total_potential_penalty_usd: Optional[float]

    top_risk_categories: List[str]
    prior_disclosure_recommended: bool
    legal_counsel_required: bool

    # Portfolio-level assessment
    overall_risk_level: str       # "critical", "high", "medium", "low", "clean"
    executive_summary: str


# ---------------------------------------------------------------------------
# Section 232 / 301 coverage (which chapters and origins are typically covered)
# ---------------------------------------------------------------------------

# Chapters typically subject to Section 232 tariffs
_SECTION_232_CHAPTERS = {
    72,  # Iron and steel
    73,  # Iron/steel articles
    74,  # Copper
    76,  # Aluminum
    82,  # Tools/cutlery of base metal
    83,  # Miscellaneous base metal articles
}

# Section 232 rates (25% steel, 10% aluminum)
_SECTION_232_RATES: Dict[int, float] = {
    72: 25.0, 73: 25.0,
    76: 10.0,
    74: 25.0, 82: 25.0, 83: 25.0,
}

# Section 301 covered origins (currently China only at 25%+ rates)
_SECTION_301_ORIGINS = {"CN", "CHN", "CHINA"}

# Chapters with known Section 301 coverage (nearly all at 25%)
_SECTION_301_HIGH_CHAPTERS = {
    39, 40, 44, 48, 61, 62, 64, 72, 73, 74, 76,
    84, 85, 87, 90, 94, 95, 96,
}

# Known circumvention-sensitive origins for China orders
_CHINA_CIRCUMVENTION_ORIGINS = {
    "VN", "TH", "MY", "TW", "MX"
}

# High-value / high-duty combinations that CBP specifically targets
_CBP_AUDIT_TRIGGERS = {
    # Chapters with known CBP audit focus
    "steel_fasteners": (73, 25.0),
    "aluminum_extrusions": (76, 10.0),
    "wooden_furniture": (94, 25.0),
    "textiles": (61, 15.0),
    "footwear": (64, 15.0),
}


# ---------------------------------------------------------------------------
# Core risk scoring functions
# ---------------------------------------------------------------------------

def _chapter_from_hts(hts_code: str) -> Optional[int]:
    if not hts_code:
        return None
    digits = re.sub(r"\D", "", hts_code)
    if len(digits) >= 2:
        try:
            return int(digits[:2])
        except ValueError:
            return None
    return None


def _estimate_penalty(
    annual_exposure_usd: Optional[float],
    severity: RiskSeverity,
) -> Tuple[Optional[float], str]:
    """Estimate penalty exposure based on CBP penalty schedule."""
    if annual_exposure_usd is None or annual_exposure_usd <= 0:
        return None, "Insufficient value data for penalty estimate"

    # Assume 2-year look-back (CBP audit typically covers last 2-3 years)
    look_back_years = 2.0
    total_underpayment = annual_exposure_usd * look_back_years

    if severity == RiskSeverity.CRITICAL:
        # Grossly negligent: 4x unpaid duty
        penalty = total_underpayment * 4.0
        basis = f"Grossly negligent penalty (4× unpaid): {look_back_years:.0f}-year look-back"
    elif severity == RiskSeverity.HIGH:
        # Negligent: 2x unpaid duty
        penalty = total_underpayment * 2.0
        basis = f"Negligent penalty (2× unpaid): {look_back_years:.0f}-year look-back"
    else:
        # Conservative estimate: 1x (underpayment only)
        penalty = total_underpayment * 1.0
        basis = f"Conservative: unpaid duty only, {look_back_years:.0f}-year look-back"

    return round(penalty, 2), basis


def score_adcvd_underpayment(
    *,
    risk_id: str,
    sku_id: Optional[str],
    description: str,
    hts_code: str,
    origin_country: str,
    ad_rate: float,
    cvd_rate: float,
    current_ad_payment: float,
    current_cvd_payment: float,
    adcvd_confidence: str,
    order_ids: List[str],
    declared_value_per_unit: Optional[float],
    annual_volume: Optional[int],
) -> Optional[PortfolioRiskFinding]:
    """Score AD/CVD underpayment risk.

    If AD/CVD applies but the company isn't paying the correct rate,
    this produces a risk finding with penalty modeling.
    """
    combined_exposure = (ad_rate - current_ad_payment) + (cvd_rate - current_cvd_payment)
    if combined_exposure <= 0.5 or adcvd_confidence == "none":
        return None

    # Compute annual exposure
    annual_exposure = None
    if declared_value_per_unit and annual_volume:
        annual_exposure = declared_value_per_unit * annual_volume * combined_exposure / 100.0

    severity = (
        RiskSeverity.CRITICAL if (adcvd_confidence == "high" and combined_exposure > 10.0) else
        RiskSeverity.HIGH if (adcvd_confidence in ("high", "medium") and combined_exposure > 5.0) else
        RiskSeverity.MEDIUM
    )

    penalty, penalty_basis = _estimate_penalty(annual_exposure, severity)

    order_str = ", ".join(order_ids[:3])
    if len(order_ids) > 3:
        order_str += f" (+{len(order_ids) - 3} more)"

    return PortfolioRiskFinding(
        risk_id=risk_id,
        sku_id=sku_id,
        description=description,
        category=RiskCategory.AD_CVD_UNDERPAYMENT,
        severity=severity,
        estimated_exposure_pct=round(combined_exposure, 4),
        estimated_annual_exposure_usd=round(annual_exposure, 2) if annual_exposure else None,
        potential_penalty_usd=penalty,
        penalty_basis=penalty_basis,
        risk_summary=(
            f"Potential AD/CVD underpayment: order(s) {order_str} may apply "
            f"({combined_exposure:.1f}% additional duty)"
        ),
        risk_detail=(
            f"Active AD/CVD order(s) [{order_str}] cover products with HTS code {hts_code} "
            f"from {origin_country}. "
            f"AD rate: {ad_rate:.1f}%, CVD rate: {cvd_rate:.1f}%. "
            f"If these rates apply and are not being paid, entries are subject to "
            f"retroactive assessment plus interest and penalties."
        ),
        immediate_actions=[
            f"Verify whether your product is within scope of order(s): {order_str}",
            "Engage licensed customs broker to assess AD/CVD liability immediately",
            "Pull all entries in the last 2 years and calculate potential assessment",
            "Consider prior disclosure to CBP if underpayment has occurred",
        ],
        remediation_steps=[
            "Request scope ruling from Commerce Department if product scope is unclear",
            "Update entry procedures to deposit correct AD/CVD duties going forward",
            "Maintain scope determination documentation for 5 years",
        ],
        legal_citations=[
            "Tariff Act of 1930 §731 (AD) / §701 (CVD)",
            "19 USC §1673 — Antidumping duties",
            "19 USC §1592 — Penalties for false statements",
            f"Active order(s): {order_str}",
        ],
        confidence=adcvd_confidence,
        requires_legal_counsel=severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH),
        prior_disclosure_recommended=(
            severity == RiskSeverity.CRITICAL and annual_exposure is not None and annual_exposure > 50000
        ),
    )


def score_section_232_risk(
    *,
    risk_id: str,
    sku_id: Optional[str],
    description: str,
    hts_code: str,
    origin_country: str,
    declared_section_232: float,
    declared_value_per_unit: Optional[float],
    annual_volume: Optional[int],
) -> Optional[PortfolioRiskFinding]:
    """Score Section 232 steel/aluminum underpayment risk.

    If the product is in a 232-covered chapter but 232 is not being paid, flag it.
    """
    chapter = _chapter_from_hts(hts_code)
    if chapter not in _SECTION_232_CHAPTERS:
        return None

    expected_rate = _SECTION_232_RATES.get(chapter, 25.0)
    exposure = expected_rate - declared_section_232

    if exposure <= 1.0:
        return None  # Already paying

    # Some origins are exempted (AU, KR, EU for steel/aluminum under agreements)
    exempted_origins = {"AU", "KR", "AR", "BR"}
    if origin_country.upper() in exempted_origins:
        return None

    annual_exposure = None
    if declared_value_per_unit and annual_volume:
        annual_exposure = declared_value_per_unit * annual_volume * exposure / 100.0

    severity = (
        RiskSeverity.HIGH if exposure > 20.0 else
        RiskSeverity.MEDIUM if exposure > 10.0 else
        RiskSeverity.LOW
    )

    penalty, penalty_basis = _estimate_penalty(annual_exposure, severity)

    material = "steel" if chapter in {72, 73} else "aluminum" if chapter == 76 else "base metal"

    return PortfolioRiskFinding(
        risk_id=risk_id,
        sku_id=sku_id,
        description=description,
        category=RiskCategory.SECTION_232_UNDERPAYMENT,
        severity=severity,
        estimated_exposure_pct=round(exposure, 4),
        estimated_annual_exposure_usd=round(annual_exposure, 2) if annual_exposure else None,
        potential_penalty_usd=penalty,
        penalty_basis=penalty_basis,
        risk_summary=(
            f"Section 232 {material} tariff may not be fully paid "
            f"(expected {expected_rate:.0f}%, showing {declared_section_232:.1f}%)"
        ),
        risk_detail=(
            f"HTS {hts_code} (chapter {chapter}) is subject to Section 232 "
            f"{material} tariffs of {expected_rate:.0f}%. "
            f"Current declared 232 rate: {declared_section_232:.1f}%. "
            f"Potential underpayment: {exposure:.1f} percentage points."
        ),
        immediate_actions=[
            f"Verify whether Section 232 {material} tariff applies to this product and origin",
            "Check if product or origin has a valid Section 232 exclusion",
            "Review entry filings for last 2 years",
        ],
        remediation_steps=[
            "Update entry procedures to include Section 232 duty",
            "Apply for Section 232 exclusion if product qualifies",
        ],
        legal_citations=[
            "19 USC §1862 — Section 232 National Security tariffs",
            f"Presidential Proclamation (Section 232, {material})",
        ],
        confidence="medium",
        requires_legal_counsel=(severity == RiskSeverity.HIGH),
    )


def score_section_301_risk(
    *,
    risk_id: str,
    sku_id: Optional[str],
    description: str,
    hts_code: str,
    origin_country: str,
    declared_section_301: float,
    declared_value_per_unit: Optional[float],
    annual_volume: Optional[int],
) -> Optional[PortfolioRiskFinding]:
    """Score Section 301 China tariff underpayment risk."""

    # 301 applies to Chinese-origin goods primarily
    origin_upper = (origin_country or "").upper()
    if origin_upper not in _SECTION_301_ORIGINS:
        return None

    chapter = _chapter_from_hts(hts_code)
    if chapter not in _SECTION_301_HIGH_CHAPTERS:
        return None

    expected_rate = 25.0  # Most 301 items are at 25%
    exposure = expected_rate - declared_section_301

    if exposure <= 1.0:
        return None  # Already paying

    annual_exposure = None
    if declared_value_per_unit and annual_volume:
        annual_exposure = declared_value_per_unit * annual_volume * exposure / 100.0

    severity = (
        RiskSeverity.HIGH if exposure > 20.0 else
        RiskSeverity.MEDIUM
    )

    penalty, penalty_basis = _estimate_penalty(annual_exposure, severity)

    return PortfolioRiskFinding(
        risk_id=risk_id,
        sku_id=sku_id,
        description=description,
        category=RiskCategory.SECTION_301_UNDERPAYMENT,
        severity=severity,
        estimated_exposure_pct=round(exposure, 4),
        estimated_annual_exposure_usd=round(annual_exposure, 2) if annual_exposure else None,
        potential_penalty_usd=penalty,
        penalty_basis=penalty_basis,
        risk_summary=(
            f"Section 301 China tariff may be underpaid "
            f"(expected {expected_rate:.0f}%, showing {declared_section_301:.1f}%)"
        ),
        risk_detail=(
            f"HTS {hts_code} from {origin_country} is typically subject to "
            f"Section 301 List 3/4 tariffs of {expected_rate:.0f}%. "
            f"Declared 301 rate: {declared_section_301:.1f}%. "
            f"Check whether this item has an active Section 301 exclusion."
        ),
        immediate_actions=[
            "Verify whether an active Section 301 exclusion covers this HTS code and origin",
            "Review USTR exclusion database for matching exclusion",
            "If no exclusion applies, assess unpaid duty liability",
        ],
        remediation_steps=[
            "Update duty payment procedures to include Section 301",
            "File for USTR exclusion if product qualifies",
            "Consider prior disclosure if material underpayment has occurred",
        ],
        legal_citations=[
            "19 USC §2411 — Section 301 trade remedies",
            "USTR Section 301 Lists 1-4 (2018-2019)",
        ],
        confidence="medium",
        requires_legal_counsel=False,
    )


def score_audit_trigger(
    *,
    risk_id: str,
    sku_id: Optional[str],
    description: str,
    hts_code: str,
    origin_country: str,
    total_effective_rate: float,
    declared_value_per_unit: Optional[float],
    annual_volume: Optional[int],
    adcvd_exposure: bool,
) -> Optional[PortfolioRiskFinding]:
    """Flag products with profiles that CBP specifically targets for audit.

    CBP's Automated Targeting System uses value, duty rate, origin, and HTS code
    to select entries for examination.  Certain combinations are red flags.
    """
    chapter = _chapter_from_hts(hts_code)
    origin_upper = (origin_country or "").upper()

    # High-rate Chinese imports with AD/CVD are a primary audit target
    if (origin_upper in _SECTION_301_ORIGINS and
            total_effective_rate >= 30.0 and
            adcvd_exposure):

        annual_exposure = None
        if declared_value_per_unit and annual_volume:
            annual_exposure = declared_value_per_unit * annual_volume * total_effective_rate / 100.0

        return PortfolioRiskFinding(
            risk_id=risk_id,
            sku_id=sku_id,
            description=description,
            category=RiskCategory.AUDIT_TRIGGER,
            severity=RiskSeverity.MEDIUM,
            estimated_exposure_pct=0.0,
            estimated_annual_exposure_usd=annual_exposure,
            potential_penalty_usd=None,
            penalty_basis="Audit trigger — no direct duty exposure estimate",
            risk_summary=(
                f"High audit risk: Chinese-origin product with {total_effective_rate:.1f}% "
                f"effective rate and potential AD/CVD exposure"
            ),
            risk_detail=(
                f"This product has a duty profile (origin: {origin_country}, "
                f"effective rate: {total_effective_rate:.1f}%, HTS: {hts_code}) "
                "that matches CBP's Automated Targeting System high-risk profile. "
                "Chinese-origin goods with combined 232/301/AD/CVD exposure frequently "
                "receive priority examination."
            ),
            immediate_actions=[
                "Ensure all entry documentation is complete and accurate",
                "Confirm classification, origin, and duty payment are correct",
                "Maintain comprehensive file for each shipment (commercial invoice, packing list, B/L, C/O)",
            ],
            remediation_steps=[
                "Consider CBP Trade Partnership Against Terrorism (C-TPAT) enrollment to reduce exam frequency",
                "Conduct internal import audit to confirm duty compliance",
                "Engage customs broker for pre-import review on high-value shipments",
            ],
            legal_citations=[
                "19 USC §1581 — CBP examination authority",
                "19 CFR Part 163 — Recordkeeping requirements",
            ],
            confidence="medium",
            requires_legal_counsel=False,
        )

    return None


def score_classification_risk(
    *,
    risk_id: str,
    sku_id: Optional[str],
    description: str,
    current_hts_code: str,
    gri_suggested_code: Optional[str],
    gri_confidence: Optional[str],
    declared_value_per_unit: Optional[float],
    annual_volume: Optional[int],
    current_total_rate: float,
) -> Optional[PortfolioRiskFinding]:
    """Flag classification risk when GRI analysis suggests a different heading."""

    if not gri_suggested_code or not current_hts_code:
        return None

    # Normalize for comparison
    current_digits = re.sub(r"\D", "", current_hts_code)[:6]
    suggested_digits = re.sub(r"\D", "", gri_suggested_code)[:6]

    if current_digits == suggested_digits:
        return None  # Same subheading — no issue

    # Heading-level disagreement (first 4 digits)
    if current_digits[:4] == suggested_digits[:4]:
        return None  # Same heading, just different subheading — lower risk

    # Different heading — potential misclassification
    if gri_confidence not in ("high", "medium"):
        return None  # Only flag when GRI is reasonably confident

    return PortfolioRiskFinding(
        risk_id=risk_id,
        sku_id=sku_id,
        description=description,
        category=RiskCategory.CLASSIFICATION_RISK,
        severity=RiskSeverity.MEDIUM,
        estimated_exposure_pct=0.0,
        estimated_annual_exposure_usd=None,
        potential_penalty_usd=None,
        penalty_basis="Classification risk — rate difference unknown without correct code",
        risk_summary=(
            f"GRI analysis suggests heading {suggested_digits[:4]} "
            f"vs declared {current_digits[:4]} — misclassification risk"
        ),
        risk_detail=(
            f"GRI analysis ({gri_confidence} confidence) suggests this product "
            f"may be more correctly classified under HTS {gri_suggested_code} "
            f"rather than the declared {current_hts_code}. "
            "Misclassification is a violation under 19 USC §1592 and can result "
            "in retroactive duty assessments and penalties."
        ),
        immediate_actions=[
            f"Compare product specification against HTS {gri_suggested_code} heading description",
            f"Compare product specification against current HTS {current_hts_code} heading description",
            "Engage licensed customs broker to confirm correct classification",
        ],
        remediation_steps=[
            "Request a CBP binding ruling to lock in correct classification",
            "If current classification is wrong, file amended entries with correct code",
            "Update classification procedures for future imports",
        ],
        legal_citations=[
            "19 USC §1592 — Penalties for false statements",
            "GRI Rules 1-6 — General Rules of Interpretation",
            "19 CFR Part 177 — CBP binding rulings",
        ],
        confidence=gri_confidence or "low",
        requires_legal_counsel=False,
    )


# ---------------------------------------------------------------------------
# Portfolio-level risk aggregation
# ---------------------------------------------------------------------------

def score_portfolio_risk(
    sku_results: List[Dict[str, Any]],
) -> Tuple[List[PortfolioRiskFinding], PortfolioRiskSummary]:
    """Score compliance risk across the entire BOM portfolio.

    Args:
        sku_results: List of per-SKU analysis result dicts (same format as
            bom_engineering_report's input).

    Returns:
        (list of PortfolioRiskFinding, PortfolioRiskSummary)
    """
    findings: List[PortfolioRiskFinding] = []
    risk_counter = [0]

    def _risk_id(sku_id: Optional[str]) -> str:
        risk_counter[0] += 1
        prefix = (sku_id or "SKU")[:8]
        return f"{prefix}-RISK-{risk_counter[0]:04d}"

    for result in sku_results:
        sku_id = result.get("sku_id")
        description = result.get("description", "")
        hts_code = result.get("current_hts_code") or ""
        origin = result.get("origin_country") or ""
        value = result.get("declared_value_per_unit")
        volume = result.get("annual_volume")
        baseline_rate = float(result.get("baseline_total_rate", 0.0))
        section_232 = float(result.get("section_232_rate", 0.0))
        section_301 = float(result.get("section_301_rate", 0.0))
        ad_rate = float(result.get("ad_duty_rate", 0.0))
        cvd_rate = float(result.get("cvd_duty_rate", 0.0))
        adcvd_confidence = result.get("adcvd_confidence", "none")
        adcvd_orders = result.get("adcvd_orders", [])
        has_adcvd = result.get("has_adcvd_exposure", False)

        # 1. AD/CVD underpayment
        # The risk is that AD/CVD exposure exists but may not be declared.
        # We pass current_ad_payment=0 to model the worst-case risk exposure —
        # the importer may not be aware of the orders or may be underpaying.
        if has_adcvd or (ad_rate + cvd_rate > 0):
            order_ids = [o.get("order_id", "?") for o in adcvd_orders] if adcvd_orders else []
            finding = score_adcvd_underpayment(
                risk_id=_risk_id(sku_id),
                sku_id=sku_id,
                description=description,
                hts_code=hts_code,
                origin_country=origin,
                ad_rate=ad_rate,
                cvd_rate=cvd_rate,
                current_ad_payment=0.0,   # Risk model: flag full potential exposure
                current_cvd_payment=0.0,
                adcvd_confidence=adcvd_confidence,
                order_ids=order_ids,
                declared_value_per_unit=value,
                annual_volume=volume,
            )
            # Only surface when confidence is high enough to warrant flagging
            if finding and adcvd_confidence in ("high", "medium") and (ad_rate + cvd_rate > 10.0):
                findings.append(finding)

        # 2. Section 232 risk
        if section_232 < _SECTION_232_RATES.get(_chapter_from_hts(hts_code) or 0, 0.0) - 1.0:
            finding = score_section_232_risk(
                risk_id=_risk_id(sku_id),
                sku_id=sku_id,
                description=description,
                hts_code=hts_code,
                origin_country=origin,
                declared_section_232=section_232,
                declared_value_per_unit=value,
                annual_volume=volume,
            )
            if finding:
                findings.append(finding)

        # 3. Section 301 risk
        origin_upper = origin.upper()
        chapter = _chapter_from_hts(hts_code) or 0
        if (origin_upper in _SECTION_301_ORIGINS and
                chapter in _SECTION_301_HIGH_CHAPTERS and
                section_301 < 20.0):  # Under the expected 25%
            finding = score_section_301_risk(
                risk_id=_risk_id(sku_id),
                sku_id=sku_id,
                description=description,
                hts_code=hts_code,
                origin_country=origin,
                declared_section_301=section_301,
                declared_value_per_unit=value,
                annual_volume=volume,
            )
            if finding:
                findings.append(finding)

        # 4. Audit trigger
        finding = score_audit_trigger(
            risk_id=_risk_id(sku_id),
            sku_id=sku_id,
            description=description,
            hts_code=hts_code,
            origin_country=origin,
            total_effective_rate=baseline_rate,
            declared_value_per_unit=value,
            annual_volume=volume,
            adcvd_exposure=has_adcvd or (ad_rate + cvd_rate > 0),
        )
        if finding:
            findings.append(finding)

        # 5. Classification risk (if GRI analysis was run)
        gri_code = result.get("gri_suggested_code")
        gri_conf = result.get("gri_confidence")
        if gri_code:
            finding = score_classification_risk(
                risk_id=_risk_id(sku_id),
                sku_id=sku_id,
                description=description,
                current_hts_code=hts_code,
                gri_suggested_code=gri_code,
                gri_confidence=gri_conf,
                declared_value_per_unit=value,
                annual_volume=volume,
                current_total_rate=baseline_rate,
            )
            if finding:
                findings.append(finding)

    # Build summary
    summary = _build_risk_summary(findings)
    return findings, summary


def _build_risk_summary(findings: List[PortfolioRiskFinding]) -> PortfolioRiskSummary:
    """Aggregate findings into a portfolio-level risk summary."""

    if not findings:
        return PortfolioRiskSummary(
            total_risk_findings=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            total_estimated_exposure_usd=None,
            total_potential_penalty_usd=None,
            top_risk_categories=[],
            prior_disclosure_recommended=False,
            legal_counsel_required=False,
            overall_risk_level="clean",
            executive_summary="No compliance risk findings identified in this portfolio.",
        )

    critical = [f for f in findings if f.severity == RiskSeverity.CRITICAL]
    high = [f for f in findings if f.severity == RiskSeverity.HIGH]
    medium = [f for f in findings if f.severity == RiskSeverity.MEDIUM]
    low = [f for f in findings if f.severity == RiskSeverity.LOW]

    total_exposure = sum(
        f.estimated_annual_exposure_usd
        for f in findings
        if f.estimated_annual_exposure_usd is not None
    )
    total_penalty = sum(
        f.potential_penalty_usd
        for f in findings
        if f.potential_penalty_usd is not None
    )

    # Category frequency
    cat_counts: Dict[str, int] = {}
    for f in findings:
        cat_counts[f.category.value] = cat_counts.get(f.category.value, 0) + 1
    top_cats = sorted(cat_counts.keys(), key=lambda c: -cat_counts[c])[:3]

    prior_disclosure = any(f.prior_disclosure_recommended for f in findings)
    legal_counsel = any(f.requires_legal_counsel for f in findings)

    if critical:
        overall = "critical"
        summary_str = (
            f"CRITICAL COMPLIANCE RISK: {len(critical)} critical finding(s) require "
            f"immediate action. Estimated annual exposure: "
            f"${total_exposure:,.0f}. Prior disclosure to CBP may be warranted."
        )
    elif high:
        overall = "high"
        summary_str = (
            f"HIGH COMPLIANCE RISK: {len(high)} high-severity finding(s) require "
            f"attention within 30 days. Estimated annual exposure: ${total_exposure:,.0f}."
        )
    elif medium:
        overall = "medium"
        summary_str = (
            f"MEDIUM COMPLIANCE RISK: {len(medium)} medium-severity finding(s). "
            "Address within 90 days to reduce audit and penalty exposure."
        )
    else:
        overall = "low"
        summary_str = (
            f"LOW COMPLIANCE RISK: {len(low)} low-severity finding(s). "
            "No immediate action required. Continue monitoring."
        )

    return PortfolioRiskSummary(
        total_risk_findings=len(findings),
        critical_count=len(critical),
        high_count=len(high),
        medium_count=len(medium),
        low_count=len(low),
        total_estimated_exposure_usd=round(total_exposure, 2) if total_exposure else None,
        total_potential_penalty_usd=round(total_penalty, 2) if total_penalty else None,
        top_risk_categories=top_cats,
        prior_disclosure_recommended=prior_disclosure,
        legal_counsel_required=legal_counsel,
        overall_risk_level=overall,
        executive_summary=summary_str,
    )
