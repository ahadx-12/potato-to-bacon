"""Core engineering opportunity model.

This is the fundamental output of a tariff engineer — not a duty rate, but an
*opportunity*: a specific, actionable finding that a company can act on to
reduce its legal duty burden.

A tariff calculator tells you what you owe. A tariff engineer tells you:
  1. What your total exposure is (base + 232 + 301 + AD/CVD, net of FTA/exclusions)
  2. Where specific savings exist and how to capture them
  3. What evidence you need to support each opportunity
  4. How confident we are in each finding
  5. What happens if you do nothing (risk of undetected AD/CVD exposure, etc.)

Every finding produced by the engine maps to one of these opportunity types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Opportunity classification
# ---------------------------------------------------------------------------

class OpportunityType(str, Enum):
    """The strategic lever being applied.

    RECLASSIFICATION
        The product is being classified under the wrong HTS heading.
        Correct classification under a different (lower-rate) heading is
        legally defensible under GRI rules.  Example: a composite product
        classified as Chapter 84 machinery that should be Chapter 85
        electrical equipment at a lower rate.

    PRODUCT_ENGINEERING
        A physical change to the product (material substitution, assembly
        sequence change, surface treatment) shifts its HTS classification
        into a lower-rate bracket.  Example: adding a felt outsole overlay
        to footwear to qualify for the textile-dominant code.

    TRADE_LANE
        A supply chain change (origin shift, assembly location, FTA
        qualification steps) reduces duty via treaty preference.
        Example: shift final assembly from CN to MX to utilize USMCA and
        eliminate the 25% Section 301 tariff.

    AD_CVD_EXPOSURE
        The product has undetected antidumping or countervailing duty
        exposure.  This is a RISK finding, not a savings opportunity — it
        means the company may be underpaying and is exposed to CBP audit.

    AD_CVD_ENGINEERING
        A targeted change (origin shift, product scope change) takes the
        product outside the scope of an active AD/CVD order.  This is
        high-complexity and always requires professional legal review.

    FTA_UTILIZATION
        The product is imported from an FTA partner country but FTA
        preference is not being claimed.  Capturing the preference
        eliminates or reduces the base duty.  Example: importing from
        Canada under USMCA but filing at MFN rates.

    EXCLUSION_FILING
        An active Section 232 or 301 exclusion covers this product and
        origin.  Filing to apply the exclusion eliminates the overlay
        tariff.  Example: a granted exclusion for a specific HTS code +
        China origin pair.

    DOCUMENTATION
        No physical or supply chain change is needed.  The savings are
        available now, but require providing additional documentation to
        CBP (fiber lab certificates, material declarations, manufacturer
        certificates of origin, etc.) to support a lower-rate claim.
        This is the lowest-risk, fastest-payback opportunity type.
    """

    RECLASSIFICATION = "reclassification"
    PRODUCT_ENGINEERING = "product_engineering"
    TRADE_LANE = "trade_lane"
    AD_CVD_EXPOSURE = "ad_cvd_exposure"
    AD_CVD_ENGINEERING = "ad_cvd_engineering"
    FTA_UTILIZATION = "fta_utilization"
    EXCLUSION_FILING = "exclusion_filing"
    DOCUMENTATION = "documentation"


class OpportunityConfidence(str, Enum):
    """How certain we are that this opportunity is real and defensible.

    HIGH   : Z3-verified, matched against real HTS data, legally clean.
    MEDIUM : Plausible but requires professional review before acting.
    LOW    : Speculative; listed for awareness, not for immediate action.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OpportunityRisk(str, Enum):
    """Audit / legal risk of pursuing this opportunity.

    A  : Minimal risk.  Standard practice, well-supported by CBP rulings.
    B  : Moderate risk.  Defensible but may require binding ruling request.
    C  : Elevated risk.  Requires pre-import ruling or legal counsel.
    """

    A = "A"
    B = "B"
    C = "C"


# ---------------------------------------------------------------------------
# Core opportunity model
# ---------------------------------------------------------------------------

class TariffEngineeringOpportunity(BaseModel):
    """A single, actionable tariff engineering finding.

    This is the central output unit of the engine.  Every mutation,
    reclassification candidate, FTA match, or AD/CVD flag is surfaced
    as a TariffEngineeringOpportunity so the consumer sees a uniform,
    ranked list of findings rather than raw duty rates.
    """

    # Identity
    opportunity_id: str = Field(
        ..., description="Unique ID for this opportunity within a report"
    )
    sku_id: Optional[str] = Field(
        default=None, description="SKU / part number this finding applies to"
    )
    opportunity_type: OpportunityType
    confidence: OpportunityConfidence
    risk_grade: OpportunityRisk = OpportunityRisk.B

    # What the finding is
    title: str = Field(
        ...,
        description=(
            "One-line title: what the opportunity is."
            " E.g. 'USMCA duty elimination for Mexican-origin steel bracket'"
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Plain-English explanation of the finding.  What is happening,"
            " why it saves money, and what needs to change."
        ),
    )

    # Current vs. target state
    current_hts_code: Optional[str] = Field(
        default=None, description="Current (baseline) HTS code or atom source ID"
    )
    target_hts_code: Optional[str] = Field(
        default=None, description="Target (optimized) HTS code or atom source ID"
    )
    current_origin: Optional[str] = Field(
        default=None, description="Current origin country (ISO 2-letter)"
    )
    target_origin: Optional[str] = Field(
        default=None, description="Target origin country for trade-lane opportunities"
    )

    # Duty rates
    baseline_total_rate: float = Field(
        ..., description="Total effective duty rate in the current state (percentage points)"
    )
    optimized_total_rate: float = Field(
        ..., description="Total effective duty rate after this opportunity is captured"
    )
    rate_reduction_pct: float = Field(
        ..., description="Reduction in percentage points (baseline_total_rate - optimized_total_rate)"
    )

    # Dollar impact (populated if declared_value_per_unit and annual_volume are known)
    declared_value_per_unit: Optional[float] = Field(
        default=None, description="Declared customs value per unit (USD)"
    )
    annual_volume: Optional[int] = Field(
        default=None, description="Annual shipment volume (units)"
    )
    savings_per_unit: Optional[float] = Field(
        default=None, description="Duty savings per unit at declared value (USD)"
    )
    annual_savings_estimate: Optional[float] = Field(
        default=None, description="Estimated annual savings if opportunity is captured (USD)"
    )

    # Implementation
    action_items: List[str] = Field(
        default_factory=list,
        description=(
            "Specific, ordered steps to capture this opportunity."
            " E.g. ['Request fiber content lab certificate from supplier',"
            " 'File amended entry with updated HTS code']"
        ),
    )
    evidence_required: List[str] = Field(
        default_factory=list,
        description=(
            "Documents and data the company must provide to support this finding."
            " E.g. ['Bill of materials showing cotton > 50% by weight',"
            " 'Manufacturer certificate of origin (USMCA Form)']"
        ),
    )
    estimated_implementation_cost: Optional[float] = Field(
        default=None,
        description="One-time cost to pursue this opportunity (USD), e.g. lab testing, legal fees",
    )
    implementation_time_days: Optional[int] = Field(
        default=None,
        description="Estimated calendar days from decision to captured savings",
    )

    # Legal grounding
    legal_basis: List[str] = Field(
        default_factory=list,
        description=(
            "Legal citations supporting this finding."
            " E.g. ['HTS Chapter 52 Note 2', 'GRI 3(b)', 'CBP Ruling NY N123456']"
        ),
    )
    requires_professional_review: bool = Field(
        default=False,
        description=(
            "True if this finding should be reviewed by a licensed customs broker"
            " or trade attorney before acting."
        ),
    )

    # Flags
    is_risk_finding: bool = Field(
        default=False,
        description=(
            "True for AD_CVD_EXPOSURE findings — these are risks (potential"
            " underpayment) rather than savings opportunities."
        ),
    )
    fact_patch: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Internal: the fact mutation that produces this opportunity",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def net_annual_savings(self) -> Optional[float]:
        """Net savings after implementation cost, first year only."""
        if self.annual_savings_estimate is None:
            return None
        cost = self.estimated_implementation_cost or 0.0
        return self.annual_savings_estimate - cost

    @property
    def payback_months(self) -> Optional[float]:
        """Months until implementation cost is recovered."""
        if not self.annual_savings_estimate or not self.estimated_implementation_cost:
            return None
        monthly = self.annual_savings_estimate / 12.0
        if monthly <= 0:
            return None
        return self.estimated_implementation_cost / monthly


# ---------------------------------------------------------------------------
# Duty exposure summary for a single SKU
# ---------------------------------------------------------------------------

class SKUDutyExposure(BaseModel):
    """Full duty burden picture for a single SKU."""

    sku_id: Optional[str] = None
    description: str
    origin_country: Optional[str] = None
    current_hts_code: Optional[str] = None
    inferred_category: Optional[str] = None

    # Duty layers
    base_rate: float = 0.0
    section_232_rate: float = 0.0
    section_301_rate: float = 0.0
    ad_duty_rate: float = 0.0
    cvd_duty_rate: float = 0.0
    exclusion_relief_rate: float = 0.0
    fta_preference_pct: float = 0.0
    total_effective_rate: float = 0.0

    # Flags
    has_trade_remedy_exposure: bool = False
    has_adcvd_exposure: bool = False
    has_fta_opportunity: bool = False
    requires_manual_review: bool = False

    # Opportunities found for this SKU
    opportunities: List[TariffEngineeringOpportunity] = Field(default_factory=list)
    best_savings_pct: float = 0.0

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Helpers: build opportunities from engine outputs
# ---------------------------------------------------------------------------

def build_opportunity_from_mutation(
    *,
    opportunity_id: str,
    sku_id: Optional[str],
    mutation_description: str,
    fact_patch: Dict[str, Any],
    baseline_total_rate: float,
    optimized_total_rate: float,
    declared_value_per_unit: Optional[float] = None,
    annual_volume: Optional[int] = None,
    verified: bool = True,
) -> TariffEngineeringOpportunity:
    """Convert a mutation result into a TariffEngineeringOpportunity."""

    rate_reduction = round(baseline_total_rate - optimized_total_rate, 4)
    savings_per_unit = None
    annual_savings = None
    if declared_value_per_unit is not None and rate_reduction > 0:
        savings_per_unit = round(declared_value_per_unit * rate_reduction / 100.0, 2)
        if annual_volume is not None:
            annual_savings = round(savings_per_unit * annual_volume, 2)

    # Determine opportunity type from fact patch contents
    opp_type = _classify_mutation_type(fact_patch, mutation_description)
    confidence = OpportunityConfidence.HIGH if verified else OpportunityConfidence.MEDIUM
    risk_grade = OpportunityRisk.B

    action_items = _default_action_items(opp_type, fact_patch)
    evidence_required = _default_evidence_requirements(opp_type, fact_patch)
    legal_basis = _default_legal_basis(opp_type)

    return TariffEngineeringOpportunity(
        opportunity_id=opportunity_id,
        sku_id=sku_id,
        opportunity_type=opp_type,
        confidence=confidence,
        risk_grade=risk_grade,
        title=_title_for_mutation(opp_type, mutation_description, rate_reduction),
        description=mutation_description,
        baseline_total_rate=round(baseline_total_rate, 4),
        optimized_total_rate=round(optimized_total_rate, 4),
        rate_reduction_pct=rate_reduction,
        declared_value_per_unit=declared_value_per_unit,
        annual_volume=annual_volume,
        savings_per_unit=savings_per_unit,
        annual_savings_estimate=annual_savings,
        action_items=action_items,
        evidence_required=evidence_required,
        legal_basis=legal_basis,
        requires_professional_review=(risk_grade == OpportunityRisk.C),
        fact_patch=fact_patch,
    )


def build_opportunity_from_adcvd_exposure(
    *,
    opportunity_id: str,
    sku_id: Optional[str],
    description: str,
    hts_code: str,
    origin_country: str,
    ad_rate: float,
    cvd_rate: float,
    order_ids: List[str],
    confidence: str = "high",
    current_total_rate: float = 0.0,
    declared_value_per_unit: Optional[float] = None,
    annual_volume: Optional[int] = None,
) -> TariffEngineeringOpportunity:
    """Build an AD/CVD exposure risk finding."""

    combined_rate = ad_rate + cvd_rate
    exposure_per_unit = None
    annual_exposure = None
    if declared_value_per_unit is not None and combined_rate > 0:
        exposure_per_unit = round(declared_value_per_unit * combined_rate / 100.0, 2)
        if annual_volume is not None:
            annual_exposure = round(exposure_per_unit * annual_volume, 2)

    order_str = ", ".join(order_ids[:3])
    if len(order_ids) > 3:
        order_str += f" (+{len(order_ids) - 3} more)"

    return TariffEngineeringOpportunity(
        opportunity_id=opportunity_id,
        sku_id=sku_id,
        opportunity_type=OpportunityType.AD_CVD_EXPOSURE,
        confidence=OpportunityConfidence(confidence) if confidence in ("high", "medium", "low") else OpportunityConfidence.MEDIUM,
        risk_grade=OpportunityRisk.A,
        title=f"AD/CVD exposure alert: {description[:60]}",
        description=(
            f"Active antidumping/countervailing duty order(s) [{order_str}] "
            f"may apply to this product from {origin_country}. "
            f"AD rate: {ad_rate}%, CVD rate: {cvd_rate}%. "
            "Undetected AD/CVD exposure creates CBP audit risk and potential duty underpayment liability."
        ),
        current_hts_code=hts_code,
        current_origin=origin_country,
        baseline_total_rate=round(current_total_rate + combined_rate, 4),
        optimized_total_rate=round(current_total_rate, 4),
        rate_reduction_pct=0.0,
        declared_value_per_unit=declared_value_per_unit,
        annual_volume=annual_volume,
        savings_per_unit=exposure_per_unit,
        annual_savings_estimate=annual_exposure,
        action_items=[
            f"Verify scope: confirm whether your product falls within the CBP scope of order(s) {order_str}",
            "Engage licensed customs broker to assess AD/CVD liability",
            "Request a scope ruling from Commerce Department if product scope is unclear",
            "Consider prior disclosure if entries have been underpaying",
        ],
        evidence_required=[
            "Product specification sheet with all materials and dimensions",
            "Production records showing manufacturing process",
            "Supplier invoices and mill certificates",
        ],
        legal_basis=[
            f"Tariff Act of 1930 §731 (AD) / §701 (CVD)",
            f"Active order(s): {order_str}",
        ],
        requires_professional_review=True,
        is_risk_finding=True,
        fact_patch=None,
    )


def build_opportunity_from_fta(
    *,
    opportunity_id: str,
    sku_id: Optional[str],
    description: str,
    hts_code: str,
    origin_country: str,
    import_country: str,
    program_id: str,
    program_name: str,
    preference_pct: float,
    base_rate: float,
    current_total_rate: float,
    declared_value_per_unit: Optional[float] = None,
    annual_volume: Optional[int] = None,
    missing_requirements: Optional[List[str]] = None,
) -> TariffEngineeringOpportunity:
    """Build an FTA utilization opportunity."""

    reduction = round(base_rate * preference_pct / 100.0, 4)
    optimized_rate = round(current_total_rate - reduction, 4)
    rate_reduction = round(current_total_rate - optimized_rate, 4)

    savings_per_unit = None
    annual_savings = None
    if declared_value_per_unit is not None and rate_reduction > 0:
        savings_per_unit = round(declared_value_per_unit * rate_reduction / 100.0, 2)
        if annual_volume is not None:
            annual_savings = round(savings_per_unit * annual_volume, 2)

    missing = missing_requirements or []
    if missing:
        conf = OpportunityConfidence.MEDIUM
        action_items = [
            f"Obtain {program_id} Certificate of Origin from supplier",
            f"Verify product satisfies {program_id} rules of origin (RVC/tariff shift)",
        ] + [f"Obtain documentation: {req}" for req in missing]
    else:
        conf = OpportunityConfidence.HIGH
        action_items = [
            f"Claim {program_id} preference on entry by adding Special Program Indicator",
            f"Obtain {program_id} Certificate of Origin from supplier",
            "Maintain origin documentation for 5 years",
        ]

    return TariffEngineeringOpportunity(
        opportunity_id=opportunity_id,
        sku_id=sku_id,
        opportunity_type=OpportunityType.FTA_UTILIZATION,
        confidence=conf,
        risk_grade=OpportunityRisk.A,
        title=f"{program_id} preference not claimed: {description[:50]}",
        description=(
            f"This product is imported from {origin_country} under {import_country}. "
            f"{program_name} ({program_id}) provides {preference_pct:.0f}% duty preference "
            f"on the base duty of {base_rate:.2f}%. "
            f"Estimated reduction: {rate_reduction:.2f} percentage points."
        ),
        current_hts_code=hts_code,
        current_origin=origin_country,
        baseline_total_rate=round(current_total_rate, 4),
        optimized_total_rate=optimized_rate,
        rate_reduction_pct=rate_reduction,
        declared_value_per_unit=declared_value_per_unit,
        annual_volume=annual_volume,
        savings_per_unit=savings_per_unit,
        annual_savings_estimate=annual_savings,
        action_items=action_items,
        evidence_required=[
            f"{program_id} Certificate of Origin (USMCA Form if USMCA)",
            "Production records demonstrating rules-of-origin compliance",
        ],
        legal_basis=[
            f"{program_name} ({program_id})",
            "19 CFR Part 182 (USMCA)" if "USMCA" in program_id else f"19 CFR Part 10 ({program_id})",
        ],
        requires_professional_review=bool(missing),
        fact_patch=None,
    )


def build_opportunity_from_exclusion(
    *,
    opportunity_id: str,
    sku_id: Optional[str],
    description: str,
    hts_code: str,
    origin_country: Optional[str],
    exclusion_id: str,
    relief_pct: float,
    current_total_rate: float,
    declared_value_per_unit: Optional[float] = None,
    annual_volume: Optional[int] = None,
) -> TariffEngineeringOpportunity:
    """Build an exclusion filing opportunity."""

    rate_reduction = min(relief_pct, current_total_rate)
    optimized_rate = round(current_total_rate - rate_reduction, 4)

    savings_per_unit = None
    annual_savings = None
    if declared_value_per_unit is not None and rate_reduction > 0:
        savings_per_unit = round(declared_value_per_unit * rate_reduction / 100.0, 2)
        if annual_volume is not None:
            annual_savings = round(savings_per_unit * annual_volume, 2)

    return TariffEngineeringOpportunity(
        opportunity_id=opportunity_id,
        sku_id=sku_id,
        opportunity_type=OpportunityType.EXCLUSION_FILING,
        confidence=OpportunityConfidence.HIGH,
        risk_grade=OpportunityRisk.A,
        title=f"Active exclusion available: {description[:50]}",
        description=(
            f"An active Section 232/301 exclusion [{exclusion_id}] covers this "
            f"product ({hts_code}) and provides {relief_pct:.1f}% tariff relief. "
            "This exclusion is not currently being claimed."
        ),
        current_hts_code=hts_code,
        current_origin=origin_country,
        baseline_total_rate=round(current_total_rate, 4),
        optimized_total_rate=optimized_rate,
        rate_reduction_pct=round(rate_reduction, 4),
        declared_value_per_unit=declared_value_per_unit,
        annual_volume=annual_volume,
        savings_per_unit=savings_per_unit,
        annual_savings_estimate=annual_savings,
        action_items=[
            f"Verify your product description matches exclusion {exclusion_id} scope",
            "File amended entries to claim the exclusion retroactively (if within 180 days)",
            "Update entry filing procedures to claim exclusion going forward",
        ],
        evidence_required=[
            "Product specification matching exclusion scope language",
            "Importer of record number and entry numbers for retroactive claims",
        ],
        legal_basis=[
            f"Section 232/301 Presidential Proclamation Exclusion: {exclusion_id}",
        ],
        requires_professional_review=False,
        fact_patch=None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_mutation_type(fact_patch: Dict[str, Any], description: str) -> OpportunityType:
    """Infer the opportunity type from a fact mutation patch."""
    if not fact_patch:
        return OpportunityType.DOCUMENTATION

    lower_desc = description.lower()
    keys = set(fact_patch.keys())

    # Origin shift -> trade lane
    if any(k.startswith("origin_country_") or k == "origin_country_raw" for k in keys):
        return OpportunityType.TRADE_LANE

    # Material change -> product engineering
    if any(k.startswith("material_") for k in keys):
        return OpportunityType.PRODUCT_ENGINEERING

    # Surface coverage change -> product engineering (physical modification)
    if any(k.startswith("surface_contact_") or k.startswith("felt_") for k in keys):
        return OpportunityType.PRODUCT_ENGINEERING

    # Construction/fiber change -> could be documentation or product
    if any(k in keys for k in ("textile_knit", "textile_woven", "fiber_cotton_dominant",
                                "fiber_polyester_dominant", "fiber_nylon_present")):
        return OpportunityType.DOCUMENTATION

    # Electronics reclassification markers
    if any(k.startswith("electronics_") for k in keys):
        if any(k in keys for k in ("electronics_insulated_conductors", "electronics_insulation_documented")):
            return OpportunityType.DOCUMENTATION
        return OpportunityType.RECLASSIFICATION

    # FTA markers
    if "fta_usmca_eligible" in keys or "assembled_in_usmca" in keys:
        return OpportunityType.TRADE_LANE

    # USMCA/FTA description keywords
    if "usmca" in lower_desc or "fta" in lower_desc or "trade lane" in lower_desc:
        return OpportunityType.TRADE_LANE

    # Default: treat as reclassification
    return OpportunityType.RECLASSIFICATION


def _title_for_mutation(opp_type: OpportunityType, description: str, rate_reduction: float) -> str:
    prefix = {
        OpportunityType.RECLASSIFICATION: "Reclassification opportunity",
        OpportunityType.PRODUCT_ENGINEERING: "Product engineering opportunity",
        OpportunityType.TRADE_LANE: "Trade lane optimization",
        OpportunityType.DOCUMENTATION: "Documentation-only quick win",
        OpportunityType.FTA_UTILIZATION: "FTA preference not claimed",
        OpportunityType.EXCLUSION_FILING: "Active exclusion not claimed",
        OpportunityType.AD_CVD_EXPOSURE: "AD/CVD exposure alert",
        OpportunityType.AD_CVD_ENGINEERING: "AD/CVD scope engineering",
    }.get(opp_type, "Tariff optimization opportunity")
    short_desc = description[:60] if len(description) > 60 else description
    return f"{prefix}: {short_desc} ({rate_reduction:.1f}pp reduction)"


def _default_action_items(opp_type: OpportunityType, fact_patch: Dict[str, Any]) -> List[str]:
    if opp_type == OpportunityType.DOCUMENTATION:
        return [
            "Gather supporting documentation from supplier",
            "File amended entry or update forward entries with documented facts",
            "Maintain documentation for 5 years per CBP requirements",
        ]
    if opp_type == OpportunityType.PRODUCT_ENGINEERING:
        return [
            "Engage product engineering team to evaluate feasibility of stated change",
            "Obtain third-party lab testing to verify modified product meets spec",
            "Submit product sample to CBP for binding ruling prior to production change",
        ]
    if opp_type == OpportunityType.TRADE_LANE:
        return [
            "Evaluate supply chain feasibility of origin shift",
            "Engage licensed customs broker to validate FTA qualification",
            "Ensure certificate of origin documentation is in place",
        ]
    if opp_type == OpportunityType.RECLASSIFICATION:
        return [
            "Review product description against all candidate HTS headings",
            "Request CBP binding ruling to lock in classification before acting",
            "Update entry filing procedures to reflect new HTS code",
        ]
    return [
        "Engage licensed customs broker or trade attorney",
        "Gather supporting product documentation",
        "File with CBP to claim savings",
    ]


def _default_evidence_requirements(opp_type: OpportunityType, fact_patch: Dict[str, Any]) -> List[str]:
    base = ["Product specification sheet"]
    if opp_type == OpportunityType.PRODUCT_ENGINEERING:
        base += [
            "Bill of materials showing proposed material changes",
            "Third-party lab test report confirming modified composition",
            "Supplier declaration for new material",
        ]
    elif opp_type == OpportunityType.DOCUMENTATION:
        base += [
            "Manufacturer's material composition certificate",
            "Third-party lab test report (if composition-dependent)",
        ]
    elif opp_type == OpportunityType.TRADE_LANE:
        base += [
            "Certificate of Origin (USMCA Form or equivalent)",
            "Manufacturing records at new origin location",
            "Regional Value Content calculation (if required)",
        ]
    elif opp_type == OpportunityType.RECLASSIFICATION:
        base += [
            "Product technical drawings and specifications",
            "Function and end-use documentation",
        ]
    return base


def _default_legal_basis(opp_type: OpportunityType) -> List[str]:
    if opp_type == OpportunityType.PRODUCT_ENGINEERING:
        return [
            "GRI Rule 1 (classification by terms of headings and notes)",
            "GRI Rule 3(b) (essential character of composite goods)",
        ]
    if opp_type == OpportunityType.DOCUMENTATION:
        return [
            "GRI Rule 1 (classification determined by heading text and chapter notes)",
            "HTS Chapter notes (material composition thresholds)",
        ]
    if opp_type == OpportunityType.TRADE_LANE:
        return [
            "19 USC §1202 (HTSUS rates of duty)",
            "USMCA / applicable FTA rules of origin",
        ]
    if opp_type == OpportunityType.RECLASSIFICATION:
        return [
            "GRI Rules 1-6",
            "CBP binding ruling authority under 19 CFR Part 177",
        ]
    return ["HTS General Rules of Interpretation"]
