"""Tariff shift and rules-of-origin analysis for FTA qualification.

FTA preference (USMCA, KORUS, CAFTA-DR, etc.) is not automatic when a product
is imported from an FTA partner country.  The product must qualify under the
applicable rules of origin, which generally require either:

  (a) A tariff shift — non-originating inputs come from different chapters,
      headings, or subheadings than the finished good (proving substantial
      transformation occurred in the FTA territory), OR

  (b) A Regional Value Content test — a minimum percentage of the product's
      value was added in the FTA territory.

  In some cases, both (a) and (b) must be satisfied.

A tariff CALCULATOR ignores this.  It checks: "is the country in the FTA?"
If yes, it reports a preference.

A tariff ENGINEER runs the actual qualification test:
  - Takes the finished good's HTS code
  - Takes the BOM with per-component origin countries and HTS codes
  - Determines which components are non-originating
  - Applies the tariff shift rule for that HTS heading
  - Reports: "qualifies" or "fails, because component X (HTS YYYY) sourced
    from CN does not satisfy the CC rule — if sourced from MX instead,
    the product would qualify and save $X/year"

This module implements that analysis.

Tariff shift rule levels (defined by FTA Annex texts):
  CC   = Change in Chapter (inputs must be from a different 2-digit chapter)
  CTH  = Change to Heading (inputs must be from a different 4-digit heading)
  CTSH = Change to Subheading (inputs must be from a different 6-digit subheading)
  CSH  = Change to Subheading (alias for CTSH)
  No Change Permitted = all inputs must already be originating
  Wholly Obtained = no non-originating inputs allowed at all

USMCA Annex 4-B and KORUS Annex 6-A contain product-specific rules (PSR).
This module includes a representative sample of rules for the most common
HTS chapters and falls back to reasonable defaults when no PSR is coded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tariff shift rule levels
# ---------------------------------------------------------------------------

class ShiftLevel:
    CC   = "CC"    # Change in Chapter (2-digit)
    CTH  = "CTH"   # Change to Heading (4-digit)
    CTSH = "CTSH"  # Change to Subheading (6-digit)
    WO   = "WO"    # Wholly Obtained — no non-originating inputs
    NA   = "NA"    # Not applicable / no preference available


@dataclass(frozen=True)
class TariffShiftRule:
    """A single product-specific tariff shift rule."""

    hts_heading_prefix: str     # 2 or 4-digit prefix this rule applies to
    shift_level: str            # CC, CTH, CTSH, WO
    rvc_required: bool          # Also requires a Regional Value Content test
    rvc_threshold_pct: float    # RVC threshold if required (e.g. 35.0)
    description: str            # Human-readable rule summary
    fta: str                    # Which FTA this rule is from
    notes: str = ""             # Additional notes or exceptions


# ---------------------------------------------------------------------------
# Representative tariff shift rules
#
# Source: USMCA Annex 4-B, KORUS Annex 6-A, representative chapters.
# This is a working subset — not the full 12,000-line Annex.
# The engine falls back to CC for unlisted chapters.
# ---------------------------------------------------------------------------

_USMCA_RULES: List[TariffShiftRule] = [
    # Chapter 39 — Plastics
    TariffShiftRule("39", ShiftLevel.CC, False, 0.0,
        "Change from any other chapter (plastics)", "USMCA"),

    # Chapter 61 — Knitted apparel
    TariffShiftRule("61", ShiftLevel.CC, True, 30.0,
        "Yarn-forward: change from fiber/yarn (chapters 50-55) + RVC 30%", "USMCA",
        notes="Yarn-forward rule requires fabric and yarn origination in USMCA territory"),

    # Chapter 62 — Woven apparel
    TariffShiftRule("62", ShiftLevel.CC, True, 30.0,
        "Yarn-forward: change from fiber/yarn + RVC 30%", "USMCA",
        notes="Yarn-forward rule"),

    # Chapter 64 — Footwear
    TariffShiftRule("64", ShiftLevel.CC, False, 0.0,
        "Change from any other chapter", "USMCA"),

    # Chapter 72-73 — Iron and Steel
    TariffShiftRule("72", ShiftLevel.CC, False, 0.0,
        "Change from any other chapter", "USMCA"),
    TariffShiftRule("73", ShiftLevel.CC, False, 0.0,
        "Change from any other chapter (steel articles)", "USMCA"),

    # Chapter 76 — Aluminum
    TariffShiftRule("76", ShiftLevel.CC, False, 0.0,
        "Change from any other chapter (aluminum articles)", "USMCA"),

    # Chapter 84 — Machinery
    TariffShiftRule("84", ShiftLevel.CTH, True, 35.0,
        "Change to heading + RVC 35% (net cost method) or 45% (transaction value)", "USMCA"),

    # Chapter 85 — Electronics
    TariffShiftRule("85", ShiftLevel.CTH, True, 35.0,
        "Change to heading + RVC 35% net cost", "USMCA"),

    # Chapter 87 — Vehicles and parts
    TariffShiftRule("8701", ShiftLevel.CC, True, 60.0,
        "Change from any other chapter + RVC 60% (passenger vehicles)", "USMCA",
        notes="Core parts list also applies — engine, transmission, etc. must be USMCA-origin"),
    TariffShiftRule("8703", ShiftLevel.CC, True, 75.0,
        "Change from any other chapter + RVC 75% net cost (passenger vehicles 2023+)", "USMCA"),
    TariffShiftRule("87", ShiftLevel.CTH, True, 35.0,
        "Change to heading + RVC 35% (automotive parts not elsewhere specified)", "USMCA"),

    # Chapter 90 — Optical and medical
    TariffShiftRule("90", ShiftLevel.CTH, False, 0.0,
        "Change to heading (optical/medical instruments)", "USMCA"),

    # Chapter 94 — Furniture
    TariffShiftRule("94", ShiftLevel.CC, False, 0.0,
        "Change from any other chapter", "USMCA"),
]

_KORUS_RULES: List[TariffShiftRule] = [
    TariffShiftRule("39", ShiftLevel.CC, False, 0.0, "Change from any other chapter", "KORUS"),
    TariffShiftRule("61", ShiftLevel.CC, True, 35.0, "Yarn-forward + RVC 35%", "KORUS"),
    TariffShiftRule("62", ShiftLevel.CC, True, 35.0, "Yarn-forward + RVC 35%", "KORUS"),
    TariffShiftRule("64", ShiftLevel.CC, False, 0.0, "Change from any other chapter", "KORUS"),
    TariffShiftRule("72", ShiftLevel.CC, False, 0.0, "Change from any other chapter", "KORUS"),
    TariffShiftRule("73", ShiftLevel.CC, False, 0.0, "Change from any other chapter", "KORUS"),
    TariffShiftRule("84", ShiftLevel.CTH, True, 35.0, "Change to heading + RVC 35%", "KORUS"),
    TariffShiftRule("85", ShiftLevel.CTH, True, 35.0, "Change to heading + RVC 35%", "KORUS"),
    TariffShiftRule("87", ShiftLevel.CTH, True, 35.0, "Change to heading + RVC 35%", "KORUS"),
    TariffShiftRule("90", ShiftLevel.CTH, False, 0.0, "Change to heading", "KORUS"),
    TariffShiftRule("94", ShiftLevel.CC, False, 0.0, "Change from any other chapter", "KORUS"),
]

# FTA partner countries (ISO 2-letter)
_FTA_PARTNERS: Dict[str, Tuple[str, List[str]]] = {
    "USMCA": ("United States-Mexico-Canada Agreement", ["MX", "CA"]),
    "KORUS": ("Korea-US Free Trade Agreement", ["KR"]),
    "CAFTA": ("Central America FTA", ["SV", "GT", "HN", "NI", "CR", "DO"]),
    "SGKFTA": ("Singapore FTA", ["SG"]),
    "AUFTA":  ("Australia FTA", ["AU"]),
    "CLFTA":  ("Chile FTA", ["CL"]),
    "PAFTA":  ("Panama FTA", ["PA"]),
    "COFTA":  ("Colombia FTA", ["CO"]),
    "PEFTA":  ("Peru FTA", ["PE"]),
    "BAFTA":  ("Bahrain FTA", ["BH"]),
    "MAFTA":  ("Morocco FTA", ["MA"]),
    "JOFTA":  ("Jordan FTA", ["JO"]),
    "ILFTA":  ("Israel FTA", ["IL"]),
    "OMFTA":  ("Oman FTA", ["OM"]),
    "GSPFTA": ("Generalized System of Preferences", [
        "IN", "BD", "PK", "LK", "TH", "ID", "PH", "VN", "ET", "TZ",
        "KE", "GH", "NG", "EG", "UA", "AM", "GE",
    ]),
}

_ALL_RULES: Dict[str, List[TariffShiftRule]] = {
    "USMCA": _USMCA_RULES,
    "KORUS": _KORUS_RULES,
}


# ---------------------------------------------------------------------------
# BOM component model
# ---------------------------------------------------------------------------

@dataclass
class BOMComponent:
    """A single component in the bill of materials."""

    description: str
    hts_code: Optional[str] = None       # Component HTS code (if known)
    origin_country: Optional[str] = None # ISO 2-letter origin
    value_usd: Optional[float] = None    # Value per unit
    is_originating: Optional[bool] = None  # Override: explicitly originating or not


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class ComponentOriginStatus:
    """FTA origin status for a single BOM component."""

    component: BOMComponent
    is_originating: bool
    reason: str       # Why originating or not
    hts_chapter: Optional[str] = None
    hts_heading: Optional[str] = None
    fails_tariff_shift: bool = False
    required_origin: Optional[str] = None  # What origin would make it originating


@dataclass
class TariffShiftResult:
    """Full tariff shift analysis result for a finished good."""

    finished_hts: str
    finished_chapter: str
    finished_heading: str
    fta: str
    fta_name: str
    rule_applied: TariffShiftRule
    qualifies: bool

    # Per-component breakdown
    component_statuses: List[ComponentOriginStatus] = field(default_factory=list)
    failing_components: List[ComponentOriginStatus] = field(default_factory=list)

    # Savings opportunity data
    rvc_pct: Optional[float] = None       # Computed RVC if applicable
    rvc_required: bool = False
    rvc_threshold_pct: float = 0.0
    rvc_qualifies: Optional[bool] = None

    # Actionable findings
    action_summary: str = ""
    what_needs_to_change: List[str] = field(default_factory=list)
    annual_savings_possible: bool = False

    legal_citations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule lookup
# ---------------------------------------------------------------------------

def _normalize_hts(code: str) -> str:
    return re.sub(r"\D", "", code or "")


def _chapter(hts: str) -> str:
    digits = _normalize_hts(hts)
    return digits[:2] if len(digits) >= 2 else ""


def _heading(hts: str) -> str:
    digits = _normalize_hts(hts)
    return digits[:4] if len(digits) >= 4 else ""


def _subheading(hts: str) -> str:
    digits = _normalize_hts(hts)
    return digits[:6] if len(digits) >= 6 else ""


def _find_rule(finished_hts: str, fta: str) -> Optional[TariffShiftRule]:
    """Find the most specific matching rule for the finished good's HTS code."""
    rules = _ALL_RULES.get(fta, [])
    finished_digits = _normalize_hts(finished_hts)

    # Try progressively less specific prefixes: 4-digit, 2-digit
    for prefix_len in (4, 2):
        prefix = finished_digits[:prefix_len]
        for rule in rules:
            rule_digits = _normalize_hts(rule.hts_heading_prefix)
            if rule_digits == prefix:
                return rule

    return None


def _default_rule(fta: str) -> TariffShiftRule:
    """Fall back to CC for unlisted chapters."""
    return TariffShiftRule(
        hts_heading_prefix="",
        shift_level=ShiftLevel.CC,
        rvc_required=False,
        rvc_threshold_pct=0.0,
        description="Default: change from any other chapter (CC)",
        fta=fta,
        notes="No product-specific rule found; CC default applied per FTA general rule",
    )


def _fta_for_origin(origin_country: str, import_country: str = "US") -> List[str]:
    """Return applicable FTAs for a given origin → import country pair."""
    origin_upper = (origin_country or "").upper()
    applicable = []
    for fta_id, (_, partners) in _FTA_PARTNERS.items():
        if origin_upper in partners:
            applicable.append(fta_id)
    return applicable


# ---------------------------------------------------------------------------
# Component qualification
# ---------------------------------------------------------------------------

def _is_component_originating(
    component: BOMComponent,
    finished_chapter: str,
    finished_heading: str,
    rule: TariffShiftRule,
    fta_partner_countries: List[str],
) -> ComponentOriginStatus:
    """Determine if a single BOM component is originating under the given rule."""

    # Explicit override
    if component.is_originating is not None:
        return ComponentOriginStatus(
            component=component,
            is_originating=component.is_originating,
            reason="Explicitly declared as originating" if component.is_originating
                   else "Explicitly declared as non-originating",
        )

    component_origin = (component.origin_country or "").upper()
    comp_hts = component.hts_code or ""
    comp_chapter = _chapter(comp_hts)
    comp_heading = _heading(comp_hts)
    comp_subheading = _subheading(comp_hts)

    # If component is from an FTA partner country, it's originating (assuming
    # it also satisfies its own origin rules — simplified here)
    if component_origin in [c.upper() for c in fta_partner_countries]:
        return ComponentOriginStatus(
            component=component,
            is_originating=True,
            reason=f"Component is from FTA partner {component_origin}",
            hts_chapter=comp_chapter,
            hts_heading=comp_heading,
        )

    # Non-originating component: check tariff shift
    if not comp_hts:
        # Cannot verify tariff shift without component HTS
        return ComponentOriginStatus(
            component=component,
            is_originating=False,
            reason=(
                "Non-originating (no HTS code provided for component "
                f"'{component.description}' from {component_origin or 'unknown'}). "
                "HTS code required to verify tariff shift."
            ),
            fails_tariff_shift=True,
        )

    # Apply shift test
    if rule.shift_level == ShiftLevel.WO:
        return ComponentOriginStatus(
            component=component,
            is_originating=False,
            reason="Rule requires wholly-obtained goods; non-originating input not permitted",
            hts_chapter=comp_chapter,
            hts_heading=comp_heading,
            fails_tariff_shift=True,
        )

    if rule.shift_level == ShiftLevel.CC:
        qualifies = comp_chapter != finished_chapter and bool(comp_chapter)
        if qualifies:
            return ComponentOriginStatus(
                component=component,
                is_originating=True,
                reason=(
                    f"CC tariff shift satisfied: component ch.{comp_chapter} → "
                    f"finished good ch.{finished_chapter}"
                ),
                hts_chapter=comp_chapter,
                hts_heading=comp_heading,
            )
        else:
            return ComponentOriginStatus(
                component=component,
                is_originating=False,
                reason=(
                    f"CC tariff shift FAILS: component ch.{comp_chapter} and "
                    f"finished good ch.{finished_chapter} are the same chapter. "
                    f"Non-originating {comp_hts} from {component_origin} "
                    "prevents FTA qualification."
                ),
                hts_chapter=comp_chapter,
                hts_heading=comp_heading,
                fails_tariff_shift=True,
                required_origin=", ".join(fta_partner_countries),
            )

    if rule.shift_level == ShiftLevel.CTH:
        qualifies = comp_heading != finished_heading and bool(comp_heading)
        if qualifies:
            return ComponentOriginStatus(
                component=component,
                is_originating=True,
                reason=(
                    f"CTH tariff shift satisfied: component heading {comp_heading} → "
                    f"finished good heading {finished_heading}"
                ),
                hts_chapter=comp_chapter,
                hts_heading=comp_heading,
            )
        else:
            return ComponentOriginStatus(
                component=component,
                is_originating=False,
                reason=(
                    f"CTH tariff shift FAILS: component heading {comp_heading} and "
                    f"finished good heading {finished_heading} are the same. "
                    f"Non-originating {comp_hts} from {component_origin}."
                ),
                hts_chapter=comp_chapter,
                hts_heading=comp_heading,
                fails_tariff_shift=True,
                required_origin=", ".join(fta_partner_countries),
            )

    if rule.shift_level in (ShiftLevel.CTSH, "CSH"):
        comp_sub = comp_subheading[:6]
        fin_sub = _subheading(finished_heading)[:6]
        qualifies = comp_sub != fin_sub and bool(comp_sub)
        if qualifies:
            return ComponentOriginStatus(
                component=component,
                is_originating=True,
                reason=f"CTSH satisfied: {comp_sub} → {fin_sub}",
                hts_chapter=comp_chapter,
                hts_heading=comp_heading,
            )
        else:
            return ComponentOriginStatus(
                component=component,
                is_originating=False,
                reason=f"CTSH FAILS: {comp_sub} same subheading as {fin_sub}",
                hts_chapter=comp_chapter,
                hts_heading=comp_heading,
                fails_tariff_shift=True,
                required_origin=", ".join(fta_partner_countries),
            )

    # Unknown shift level — treat as failing
    return ComponentOriginStatus(
        component=component,
        is_originating=False,
        reason=f"Unknown tariff shift level {rule.shift_level}",
        fails_tariff_shift=True,
    )


# ---------------------------------------------------------------------------
# RVC computation
# ---------------------------------------------------------------------------

def _compute_rvc(
    component_statuses: List[ComponentOriginStatus],
    total_value: Optional[float],
    method: str = "net_cost",
) -> Optional[float]:
    """Compute Regional Value Content percentage.

    RVC (Net Cost) = (NC - VNM) / NC × 100
    where NC = net cost of finished good, VNM = value of non-originating materials.

    Simplified: if component values are declared, compute RVC directly.
    Returns None if insufficient value data.
    """
    if total_value is None or total_value <= 0:
        return None

    vnm = sum(
        (s.component.value_usd or 0.0)
        for s in component_statuses
        if not s.is_originating and s.component.value_usd is not None
    )

    rvc = ((total_value - vnm) / total_value) * 100.0
    return round(rvc, 2)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def evaluate_tariff_shift(
    finished_hts: str,
    bom_components: List[BOMComponent],
    fta: str = "USMCA",
    finished_good_value: Optional[float] = None,
) -> TariffShiftResult:
    """Evaluate whether a finished good qualifies for FTA preference.

    This is the core analysis a tariff engineer performs before recommending
    FTA utilization.  Without it, an FTA recommendation is noise.

    Args:
        finished_hts: HTS code of the finished good.
        bom_components: List of BOM components with origin and HTS code.
        fta: FTA to evaluate (default: "USMCA").
        finished_good_value: Total value per unit for RVC computation.

    Returns:
        TariffShiftResult with qualification status and actionable findings.
    """
    finished_chapter = _chapter(finished_hts)
    finished_heading = _heading(finished_hts)

    fta_name, fta_partners = _FTA_PARTNERS.get(fta, (fta, []))

    rule = _find_rule(finished_hts, fta) or _default_rule(fta)

    # Evaluate each component
    component_statuses: List[ComponentOriginStatus] = []
    for comp in bom_components:
        status = _is_component_originating(
            comp, finished_chapter, finished_heading, rule, fta_partners
        )
        component_statuses.append(status)

    failing = [s for s in component_statuses if not s.is_originating]

    # RVC computation (if rule requires it)
    rvc_pct = None
    rvc_qualifies = None
    if rule.rvc_required and finished_good_value:
        rvc_pct = _compute_rvc(component_statuses, finished_good_value)
        if rvc_pct is not None:
            rvc_qualifies = rvc_pct >= rule.rvc_threshold_pct

    # Determine overall qualification
    tariff_shift_qualifies = len(failing) == 0
    qualifies = tariff_shift_qualifies and (
        not rule.rvc_required or rvc_qualifies is not False
    )

    # Build action items
    what_needs_to_change: List[str] = []
    if not tariff_shift_qualifies:
        for s in failing:
            comp_origin = s.component.origin_country or "unknown"
            comp_desc = s.component.description[:50]
            comp_hts = s.component.hts_code or "HTS unknown"
            if s.required_origin:
                what_needs_to_change.append(
                    f"Re-source '{comp_desc}' ({comp_hts}) from {comp_origin} to "
                    f"an FTA partner ({s.required_origin}) to satisfy the "
                    f"{rule.shift_level} tariff shift rule."
                )
            else:
                what_needs_to_change.append(
                    f"'{comp_desc}' ({comp_hts}) from {comp_origin}: {s.reason}"
                )

    if rule.rvc_required and rvc_qualifies is False:
        what_needs_to_change.append(
            f"RVC requirement not met: computed {rvc_pct:.1f}% < "
            f"required {rule.rvc_threshold_pct:.0f}%. Increase value-added in "
            f"{fta_name} territory."
        )

    if qualifies:
        action_summary = (
            f"Product qualifies for {fta} preference under {rule.shift_level} rule. "
            f"Claim SPI '{fta[:2]}' on entry to obtain preference."
        )
    elif what_needs_to_change:
        action_summary = (
            f"Product DOES NOT qualify for {fta} preference. "
            f"{len(failing)} component(s) fail the {rule.shift_level} tariff shift. "
            "See what_needs_to_change for specific supply chain changes required."
        )
    else:
        action_summary = f"Product does not qualify for {fta} preference."

    legal_citations = [
        f"{fta_name} ({fta})",
        f"{fta} Annex — Product-Specific Rule: {rule.description}",
        "19 CFR Part 182 (USMCA)" if fta == "USMCA" else f"19 CFR Part 10 ({fta})",
    ]
    if rule.notes:
        legal_citations.append(f"Note: {rule.notes}")

    return TariffShiftResult(
        finished_hts=finished_hts,
        finished_chapter=finished_chapter,
        finished_heading=finished_heading,
        fta=fta,
        fta_name=fta_name,
        rule_applied=rule,
        qualifies=qualifies,
        component_statuses=component_statuses,
        failing_components=failing,
        rvc_pct=rvc_pct,
        rvc_required=rule.rvc_required,
        rvc_threshold_pct=rule.rvc_threshold_pct,
        rvc_qualifies=rvc_qualifies,
        action_summary=action_summary,
        what_needs_to_change=what_needs_to_change,
        annual_savings_possible=bool(failing),  # Savings possible if we fix the fails
        legal_citations=legal_citations,
    )


def check_fta_eligibility(
    finished_hts: str,
    origin_country: str,
    import_country: str = "US",
    bom_components: Optional[List[BOMComponent]] = None,
    finished_good_value: Optional[float] = None,
) -> Optional[TariffShiftResult]:
    """High-level eligibility check: find applicable FTAs and run tariff shift.

    Returns the best TariffShiftResult (qualifying or best-opportunity non-qualifying),
    or None if no FTA applies to this origin-import country pair.
    """
    applicable_ftas = _fta_for_origin(origin_country, import_country)
    if not applicable_ftas:
        return None

    results: List[TariffShiftResult] = []
    for fta in applicable_ftas:
        result = evaluate_tariff_shift(
            finished_hts=finished_hts,
            bom_components=bom_components or [],
            fta=fta,
            finished_good_value=finished_good_value,
        )
        results.append(result)

    # Prefer qualifying results; among non-qualifying, prefer fewest failing components
    qualifying = [r for r in results if r.qualifies]
    if qualifying:
        return qualifying[0]

    results.sort(key=lambda r: len(r.failing_components))
    return results[0] if results else None
