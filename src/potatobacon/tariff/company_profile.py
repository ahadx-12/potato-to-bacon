"""Company profile model for tariff engineering engagements.

A generic tariff analysis is necessarily hedged: "you might qualify for USMCA
if you can source components from Mexico" covers a company that has a factory
in Monterrey and a company whose entire supply chain is locked in China equally.
They need completely different advice.

A company profile captures the constraints and context that make recommendations
specific:

  - Which FTAs is the company already claiming?  (No point surfacing USMCA if
    they already use it.)
  - What are their fixed origins?  (Cannot surface trade lane opportunities for
    origins the company cannot change.)
  - Do they have existing binding rulings?  (Classification findings for those
    codes are already locked in.)
  - What are their supply chain constraints?  (Cannot recommend product
    engineering if the product is FDA-certified and cannot be changed.)
  - What is their audit risk profile?  (Companies under active CBP audit need
    different advice than those that have never been audited.)

The engineering engine uses the company profile to:
  1. Filter out infeasible opportunity types
  2. Prioritize findings based on actual constraints
  3. Personalize action items ("you already import from MX — USMCA qualification
     requires only adding a Certificate of Origin")
  4. Set the risk tolerance for opportunity grading

The profile is optional.  If not provided, the engine uses conservative defaults
and hedges every recommendation.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskTolerance(str, Enum):
    """How aggressively the company wants to pursue borderline opportunities."""

    LOW      = "low"      # Only high-confidence, Grade A opportunities
    MODERATE = "moderate" # High and medium confidence, Grade A and B
    HIGH     = "high"     # All findings, including speculative Grade C


class AuditStatus(str, Enum):
    """Current CBP audit / compliance status."""

    NONE        = "none"        # No audit history or active audit
    PRIOR       = "prior"       # Had a prior audit (now resolved)
    ACTIVE      = "active"      # Currently under CBP audit
    SETTLEMENT  = "settlement"  # In settlement / prior disclosure


class SupplyChainConstraint(str, Enum):
    """Constraints that make certain opportunity types infeasible."""

    FIXED_ORIGIN          = "fixed_origin"           # Cannot change country of origin
    FIXED_MATERIALS       = "fixed_materials"         # Cannot change material composition
    CERTIFIED_PRODUCT     = "certified_product"       # UL/CE/FDA certified, cannot change
    SINGLE_SUPPLIER       = "single_supplier"         # Sole-source, cannot re-source
    LONG_TERM_CONTRACT    = "long_term_contract"      # Locked into supply contract
    FIXED_ASSEMBLY_LOCATION = "fixed_assembly_location"  # Cannot move assembly


class CompanyProfile(BaseModel):
    """Tariff engineering company profile.

    Captures the company-level context that shapes which recommendations are
    feasible, which are already captured, and which carry elevated risk.

    All fields are optional.  Omitting a field results in conservative defaults.
    """

    # Identity
    company_name: Optional[str] = Field(
        default=None,
        description="Legal entity name (for report labeling)",
    )
    importer_of_record: Optional[str] = Field(
        default=None,
        description="IOR number or identifier",
    )

    # FTA programs
    active_fta_programs: List[str] = Field(
        default_factory=list,
        description=(
            "FTA / preference programs the company is currently claiming "
            "(e.g. ['USMCA', 'GSP']). These are NOT surfaced as new opportunities."
        ),
    )
    fta_programs_of_interest: List[str] = Field(
        default_factory=list,
        description=(
            "FTA programs the company wants to evaluate even if not currently claimed."
        ),
    )

    # Origin and supply chain constraints
    fixed_origin_countries: List[str] = Field(
        default_factory=list,
        description=(
            "ISO 2-letter origin countries that CANNOT be changed. "
            "Trade lane / USMCA opportunities for these origins are suppressed."
        ),
    )
    primary_origin_countries: List[str] = Field(
        default_factory=list,
        description=(
            "Countries the company currently sources from "
            "(used to prioritize relevant findings)."
        ),
    )
    primary_import_countries: List[str] = Field(
        default_factory=list,
        description="Countries the company imports into (default: US).",
    )
    supply_chain_constraints: List[SupplyChainConstraint] = Field(
        default_factory=list,
        description="Constraints that limit feasible opportunity types.",
    )

    # Existing CBP classifications
    existing_binding_rulings: List[str] = Field(
        default_factory=list,
        description=(
            "CBP binding ruling numbers (HQ or NY series) already held. "
            "Products covered by these rulings will not receive classification findings."
        ),
    )
    existing_hts_codes: List[str] = Field(
        default_factory=list,
        description=(
            "HTS codes the company currently files under. "
            "Reclassification findings are scoped to these codes."
        ),
    )

    # Audit and compliance
    audit_status: AuditStatus = Field(
        default=AuditStatus.NONE,
        description="Current CBP audit or compliance status.",
    )
    has_prior_disclosure: bool = Field(
        default=False,
        description="Has the company ever filed a CBP prior disclosure?",
    )

    # Risk preferences
    risk_tolerance: RiskTolerance = Field(
        default=RiskTolerance.MODERATE,
        description=(
            "How aggressively to surface borderline opportunities. "
            "LOW = Grade A only; MODERATE = A+B; HIGH = all findings."
        ),
    )
    suppress_professional_review_required: bool = Field(
        default=False,
        description=(
            "If True, suppress opportunities that require professional review "
            "(Grade C). Default False."
        ),
    )

    # Engagement context
    annual_import_value_usd: Optional[float] = Field(
        default=None,
        description="Total annual customs value (USD) — used to prioritize portfolio.",
    )
    primary_hs_chapters: List[int] = Field(
        default_factory=list,
        description=(
            "HTS chapters of the company's primary products. "
            "Used to focus chapter-specific analysis."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------
    # Feasibility helpers used by the engineering engine
    # ------------------------------------------------------------------

    def trade_lane_feasible(self, origin_country: str) -> bool:
        """Can we recommend a trade lane change for this origin?

        An origin is infeasible for trade lane opportunities when:
        - It is explicitly listed in fixed_origin_countries, OR
        - FIXED_ORIGIN constraint is set AND no specific list is provided
          (meaning ALL origins are fixed — e.g. the company is sole-sourced)
        """
        if origin_country.upper() in [c.upper() for c in self.fixed_origin_countries]:
            return False
        if (SupplyChainConstraint.FIXED_ORIGIN in self.supply_chain_constraints
                and not self.fixed_origin_countries):
            return False
        return True

    def product_engineering_feasible(self) -> bool:
        """Can we recommend product material/design changes?"""
        if SupplyChainConstraint.FIXED_MATERIALS in self.supply_chain_constraints:
            return False
        if SupplyChainConstraint.CERTIFIED_PRODUCT in self.supply_chain_constraints:
            return False
        return True

    def fta_already_claimed(self, program_id: str) -> bool:
        """Is this FTA preference already being claimed?"""
        return program_id.upper() in [p.upper() for p in self.active_fta_programs]

    def should_surface_opportunity(
        self,
        risk_grade: str,
        requires_professional_review: bool = False,
    ) -> bool:
        """Filter an opportunity by the company's risk tolerance."""
        if requires_professional_review and self.suppress_professional_review_required:
            return False
        if self.risk_tolerance == RiskTolerance.LOW and risk_grade not in ("A",):
            return False
        if self.risk_tolerance == RiskTolerance.MODERATE and risk_grade == "C":
            return False
        return True

    def is_origin_of_interest(self, origin_country: str) -> bool:
        """Is this origin country in the company's supply chain?"""
        if not self.primary_origin_countries:
            return True  # No constraint — all origins relevant
        return origin_country.upper() in [c.upper() for c in self.primary_origin_countries]


# ---------------------------------------------------------------------------
# Default profile (no constraints)
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = CompanyProfile()
