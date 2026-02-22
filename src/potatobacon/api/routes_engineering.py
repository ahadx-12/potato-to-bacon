"""Tariff Engineering API — primary interface for BOM-level analysis.

This is the core workflow endpoint for a tariff engineering engagement:

  POST /v1/engineering/analyze-bom
    A company uploads their BOM (CSV, JSON, or XLSX).
    The engine classifies every line item, computes its full duty burden
    (base + 232 + 301 + AD/CVD, net of FTA preferences and exclusions),
    discovers savings opportunities across the portfolio, and returns a
    ranked engineering report.

  POST /v1/engineering/analyze-sku
    Single-SKU version of the above — for spot-checking an individual
    product without uploading a full BOM.

The output is a BOMEngineeringReport, not a tariff dossier.
The distinction matters:
  - A dossier tells you what your duty rate is.
  - An engineering report tells you what to DO about it.
"""

from __future__ import annotations

import io
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, get_registry, resolve_tenant_from_request
from potatobacon.tariff.bom_engineering_report import BOMEngineeringReport, build_bom_engineering_report
from potatobacon.tariff.bom_parser import parse_bom_file
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.duty_calculator import compute_total_duty
from potatobacon.tariff.engine import compute_duty_result
from potatobacon.tariff.engineering_opportunity import (
    OpportunityType,
    TariffEngineeringOpportunity,
    build_opportunity_from_fta,
)
from potatobacon.tariff.fact_compiler import compile_facts
from potatobacon.tariff.fact_vocabulary import expand_facts
from potatobacon.tariff.chapter_filter import filter_atoms_by_chapter
from potatobacon.tariff.models import TariffScenario
from potatobacon.tariff.mutation_engine import MutationEngine
from potatobacon.tariff.origin_engine import build_origin_policy_atoms
from potatobacon.tariff.product_schema import ProductCategory, ProductSpecModel
from potatobacon.tariff.adcvd_registry import get_adcvd_registry
from potatobacon.tariff.fta_engine import get_fta_engine
from potatobacon.tariff.exclusion_tracker import get_exclusion_tracker
from potatobacon.tariff.hts_search import search_hts_by_description, top_chapters_for_description
from potatobacon.tariff.gri_engine import apply_gri, gri_legal_basis
from potatobacon.tariff.company_profile import CompanyProfile, DEFAULT_PROFILE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/engineering", tags=["engineering"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SKUAnalysisRequest(BaseModel):
    """Input for single-SKU engineering analysis."""

    description: str = Field(..., min_length=3)
    part_id: Optional[str] = None
    origin_country: Optional[str] = None
    import_country: Optional[str] = Field(default="US")
    hts_hint: Optional[str] = Field(
        default=None,
        description="Known or suspected HTS code (6/8/10-digit) — used to improve accuracy",
    )
    declared_value_per_unit: Optional[float] = Field(default=None, ge=0.0)
    annual_volume: Optional[int] = Field(default=None, ge=0)
    materials: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description='Material list: [{"component": "housing", "material": "steel"}, ...]',
    )
    product_category: Optional[str] = Field(
        default=None,
        description="Override inferred category (electronics/footwear/fastener/etc.)",
    )
    law_context: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class OpportunityOut(BaseModel):
    """Serializable opportunity for API response."""

    opportunity_id: str
    sku_id: Optional[str]
    opportunity_type: str
    confidence: str
    risk_grade: str
    title: str
    description: str
    baseline_total_rate: float
    optimized_total_rate: float
    rate_reduction_pct: float
    annual_savings_estimate: Optional[float]
    savings_per_unit: Optional[float]
    action_items: List[str]
    evidence_required: List[str]
    legal_basis: List[str]
    requires_professional_review: bool
    is_risk_finding: bool
    current_hts_code: Optional[str]
    target_hts_code: Optional[str]
    current_origin: Optional[str]

    model_config = ConfigDict(extra="forbid")


class SKUExposureOut(BaseModel):
    """Per-SKU duty exposure summary for API response."""

    sku_id: Optional[str]
    description: str
    origin_country: Optional[str]
    current_hts_code: Optional[str]
    inferred_category: Optional[str]
    base_rate: float
    section_232_rate: float
    section_301_rate: float
    ad_duty_rate: float
    cvd_duty_rate: float
    exclusion_relief_rate: float
    fta_preference_pct: float
    total_effective_rate: float
    has_trade_remedy_exposure: bool
    has_adcvd_exposure: bool
    opportunities: List[OpportunityOut]
    best_savings_pct: float

    model_config = ConfigDict(extra="forbid")


class PortfolioSummaryOut(BaseModel):
    total_skus: int
    skus_analyzed: int
    skus_with_opportunities: int
    skus_with_adcvd_exposure: int
    total_annual_duty_exposure: Optional[float]
    achievable_annual_savings: Optional[float]
    weighted_avg_baseline_rate: Optional[float]
    weighted_avg_optimized_rate: Optional[float]
    documentation_only_count: int
    product_engineering_count: int
    trade_lane_count: int
    reclassification_count: int
    fta_utilization_count: int
    exclusion_filing_count: int
    adcvd_exposure_count: int

    model_config = ConfigDict(extra="forbid")


class RiskFindingOut(BaseModel):
    """Serializable compliance risk finding for API response."""

    risk_id: str
    sku_id: Optional[str]
    description: str
    category: str
    severity: str
    estimated_exposure_pct: float
    estimated_annual_exposure_usd: Optional[float]
    potential_penalty_usd: Optional[float]
    penalty_basis: str
    risk_summary: str
    risk_detail: str
    immediate_actions: List[str]
    remediation_steps: List[str]
    legal_citations: List[str]
    confidence: str
    requires_legal_counsel: bool
    prior_disclosure_recommended: bool

    model_config = ConfigDict(extra="forbid")


class RiskSummaryOut(BaseModel):
    """Portfolio-level risk summary for API response."""

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
    overall_risk_level: str
    executive_summary: str

    model_config = ConfigDict(extra="forbid")


class BOMEngineeringReportOut(BaseModel):
    """Full engineering report response."""

    report_id: str
    law_context: str
    tariff_manifest_hash: str
    tenant_id: Optional[str]
    analyzed_at: str
    portfolio_summary: PortfolioSummaryOut
    all_opportunities: List[OpportunityOut]
    quick_wins: List[OpportunityOut]
    sku_findings: List[SKUExposureOut]
    risk_findings: List[RiskFindingOut]
    risk_summary: Optional[RiskSummaryOut]
    warnings: List[str]

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Core analysis helper
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, ProductCategory] = {
    "shoe": ProductCategory.FOOTWEAR,
    "sneaker": ProductCategory.FOOTWEAR,
    "footwear": ProductCategory.FOOTWEAR,
    "boot": ProductCategory.FOOTWEAR,
    "sandal": ProductCategory.FOOTWEAR,
    "bolt": ProductCategory.FASTENER,
    "screw": ProductCategory.FASTENER,
    "fastener": ProductCategory.FASTENER,
    "nut": ProductCategory.FASTENER,
    "washer": ProductCategory.FASTENER,
    "rivet": ProductCategory.FASTENER,
    "pcb": ProductCategory.ELECTRONICS,
    "circuit": ProductCategory.ELECTRONICS,
    "battery": ProductCategory.ELECTRONICS,
    "cable": ProductCategory.ELECTRONICS,
    "router": ProductCategory.ELECTRONICS,
    "motor": ProductCategory.ELECTRONICS,
    "electronics": ProductCategory.ELECTRONICS,
    "wiring": ProductCategory.ELECTRONICS,
    "harness": ProductCategory.ELECTRONICS,
    "charger": ProductCategory.ELECTRONICS,
    "connector": ProductCategory.ELECTRONICS,
    "shirt": ProductCategory.APPAREL_TEXTILE,
    "garment": ProductCategory.APPAREL_TEXTILE,
    "apparel": ProductCategory.APPAREL_TEXTILE,
    "jacket": ProductCategory.APPAREL_TEXTILE,
    "textile": ProductCategory.TEXTILE,
    "fabric": ProductCategory.TEXTILE,
    "cotton": ProductCategory.TEXTILE,
    "polyester": ProductCategory.TEXTILE,
    "chair": ProductCategory.FURNITURE,
    "desk": ProductCategory.FURNITURE,
    "furniture": ProductCategory.FURNITURE,
    "sofa": ProductCategory.FURNITURE,
    "table": ProductCategory.FURNITURE,
    "seat": ProductCategory.FURNITURE,
}


def _infer_category(description: str, materials: Optional[List[Dict[str, str]]] = None) -> ProductCategory:
    lower = description.lower()
    for keyword, category in _CATEGORY_KEYWORDS.items():
        if keyword in lower:
            return category
    if materials:
        for m in materials:
            mat_lower = (m.get("material") or "").lower()
            if "steel" in mat_lower or "aluminum" in mat_lower:
                return ProductCategory.FASTENER
            if "textile" in mat_lower or "fabric" in mat_lower:
                return ProductCategory.TEXTILE
    return ProductCategory.OTHER


def _normalize_hts_code(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) >= 10:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}.{digits[8:10]}"
    if len(digits) >= 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}"
    if len(digits) >= 6:
        return f"{digits[:4]}.{digits[4:6]}"
    return digits


def _best_hts_code_from_atoms(active_atoms: List[Any]) -> str:
    for atom in active_atoms:
        metadata = getattr(atom, "metadata", None)
        if isinstance(metadata, dict):
            hts_code = metadata.get("hts_code")
            if isinstance(hts_code, str) and hts_code.strip():
                return _normalize_hts_code(hts_code)
        source_id = getattr(atom, "source_id", "")
        if isinstance(source_id, str) and source_id.startswith("HTS_"):
            normalized = _normalize_hts_code(source_id[4:])
            if normalized:
                return normalized
    return ""


def _analyze_single_sku(
    *,
    description: str,
    part_id: Optional[str],
    origin_country: Optional[str],
    import_country: str,
    hts_hint: Optional[str],
    declared_value_per_unit: Optional[float],
    annual_volume: Optional[int],
    materials_list: Optional[List[Dict[str, str]]],
    product_category_override: Optional[str],
    atoms: List[Any],
    context_meta: Dict[str, Any],
    max_mutations: int = 10,
) -> Dict[str, Any]:
    """Run full tariff engineering analysis on a single SKU.

    Returns a dict with all fields needed by build_bom_engineering_report.
    """
    context_id = context_meta["context_id"]
    duty_rates = context_meta.get("duty_rates") or {}
    manifest_hash = context_meta.get("manifest_hash", "")

    # Build ProductSpecModel
    if product_category_override:
        try:
            category = ProductCategory(product_category_override)
        except ValueError:
            category = _infer_category(description, materials_list)
    else:
        category = _infer_category(description, materials_list)

    materials = materials_list or []
    product_spec = ProductSpecModel(
        product_category=category,
        materials=[{"component": m.get("component", "body"), "material": m.get("material", "unknown")} for m in materials],
        origin_country=origin_country,
        import_country=import_country,
        declared_value_per_unit=declared_value_per_unit,
        annual_volume=annual_volume,
    )

    # Compile and expand facts
    facts, _ = compile_facts(product_spec)
    facts = expand_facts(facts)

    # Auto-classify with HTS text search when no hts_hint is provided.
    # This makes the engine general-purpose: any product description can be
    # routed to the right HTS chapter without requiring a pre-known HTS code.
    gri_classification = None
    if hts_hint:
        hint_norm = _normalize_hts_code(hts_hint)
        if hint_norm:
            facts["hts_code"] = hint_norm
            chapter_digits = "".join(ch for ch in hint_norm if ch.isdigit())[:2]
            if chapter_digits:
                facts[f"chapter_{chapter_digits}"] = True
    else:
        try:
            hts_candidates = search_hts_by_description(description, top_n=5)
            if hts_candidates:
                gri_classification = apply_gri(description, materials_list or [], hts_candidates)
                if gri_classification.winning_heading:
                    ch = str(gri_classification.winning_chapter).zfill(2)
                    facts[f"chapter_{ch}"] = True
                    facts["hts_code"] = gri_classification.winning_code
        except Exception as exc:
            logger.debug("HTS auto-classify failed for %r: %s", description[:40], exc)

    # Chapter pre-filter
    filtered_atoms = filter_atoms_by_chapter(atoms, facts, context_id=context_id)
    if not filtered_atoms:
        filtered_atoms = atoms

    # Baseline classification
    baseline = TariffScenario(name="baseline", facts=deepcopy(facts))
    baseline_result = compute_duty_result(filtered_atoms, baseline, duty_rates=duty_rates)
    baseline_rate = baseline_result.duty_rate if baseline_result.duty_rate is not None else 0.0
    baseline_codes = [a.source_id for a in baseline_result.active_atoms]
    hts_code = hts_hint or _best_hts_code_from_atoms(baseline_result.active_atoms)
    if hts_hint:
        hts_code = _normalize_hts_code(hts_hint)

    # Full duty breakdown (base + 232 + 301 + AD/CVD + FTA + exclusions)
    try:
        baseline_breakdown = compute_total_duty(
            base_rate=baseline_rate,
            hts_code=hts_code,
            origin_country=origin_country or "",
            import_country=import_country,
            facts=facts,
            active_codes=baseline_codes,
        )
        baseline_total = baseline_breakdown.total_duty_rate
        section_232 = baseline_breakdown.section_232_rate
        section_301 = baseline_breakdown.section_301_rate
        ad_rate = baseline_breakdown.ad_duty_rate
        cvd_rate = baseline_breakdown.cvd_duty_rate
        exclusion_relief = baseline_breakdown.exclusion_relief_rate
        fta_pct = baseline_breakdown.fta_preference_pct
        base_rate = baseline_breakdown.base_rate

        # Build adcvd_orders list for report
        adcvd_orders_for_report: List[Dict[str, Any]] = []
        if baseline_breakdown.adcvd_result and baseline_breakdown.adcvd_result.has_exposure:
            for match in baseline_breakdown.adcvd_result.order_matches:
                adcvd_orders_for_report.append({
                    "order_id": match.order.order_id,
                    "order_type": match.order.order_type,
                    "duty_rate_pct": match.order.duty_rate_pct,
                    "confidence": match.confidence,
                })

        # Build fta_result dict for report
        fta_result_for_report: Optional[Dict[str, Any]] = None
        if baseline_breakdown.fta_result:
            fta = baseline_breakdown.fta_result
            best_prog = fta.best_program
            fta_result_for_report = {
                "has_eligible_program": fta.has_eligible_program,
                "best_preference_pct": fta.best_preference_pct,
                "best_program": {
                    "program_id": best_prog.program_id,
                    "program_name": best_prog.program_name,
                    "preference_pct": best_prog.preference_pct,
                    "missing_requirements": list(best_prog.missing_requirements),
                } if best_prog else None,
            }

        # Build exclusion_result dict for report
        excl_result_for_report: Optional[Dict[str, Any]] = None
        if baseline_breakdown.exclusion_result:
            er = baseline_breakdown.exclusion_result
            excl_result_for_report = {
                "has_active_exclusion": er.has_active_exclusion,
                "total_exclusion_relief_pct": er.total_exclusion_relief_pct,
                "exclusion_id": getattr(er, "exclusion_id", ""),
            }

        adcvd_confidence = (
            baseline_breakdown.adcvd_result.confidence
            if baseline_breakdown.adcvd_result else "none"
        )
    except Exception as exc:
        logger.warning("Duty breakdown failed for %s: %s", description[:40], exc)
        baseline_total = baseline_rate
        section_232 = 0.0
        section_301 = 0.0
        ad_rate = 0.0
        cvd_rate = 0.0
        exclusion_relief = 0.0
        fta_pct = 0.0
        base_rate = baseline_rate
        adcvd_orders_for_report = []
        fta_result_for_report = None
        excl_result_for_report = None
        adcvd_confidence = "none"

    # Discover optimization mutations
    mutation_engine = MutationEngine(filtered_atoms, duty_rates=duty_rates)
    derived_mutations = mutation_engine.discover_mutations(
        baseline, baseline_rate, max_candidates=max_mutations
    )

    mutation_results: List[Dict[str, Any]] = []
    for dm in derived_mutations:
        mutated = TariffScenario(
            name="mutation",
            facts={**deepcopy(facts), **dm.fact_patch},
        )
        mut_result = compute_duty_result(filtered_atoms, mutated, duty_rates=duty_rates)
        mut_rate = mut_result.duty_rate if mut_result.duty_rate is not None else baseline_rate
        mut_codes = [a.source_id for a in mut_result.active_atoms]
        mut_hts = hts_code or _best_hts_code_from_atoms(mut_result.active_atoms)

        try:
            mut_breakdown = compute_total_duty(
                base_rate=mut_rate,
                hts_code=mut_hts,
                origin_country=str(mutated.facts.get("origin_country_raw") or origin_country or ""),
                import_country=import_country,
                facts=mutated.facts,
                active_codes=mut_codes,
            )
            mut_effective = mut_breakdown.total_duty_rate
        except Exception:
            mut_effective = mut_rate

        effective_savings = baseline_total - mut_effective
        if effective_savings <= 0:
            continue

        gri_basis = gri_legal_basis(gri_classification) if gri_classification else []
        mutation_results.append({
            "human_description": dm.human_description,
            "fact_patch": dm.fact_patch,
            "projected_duty_rate": mut_rate,
            "effective_duty_rate": mut_effective,
            "savings_vs_baseline": baseline_rate - mut_rate,
            "effective_savings": effective_savings,
            "verified": dm.verified,
            "gri_legal_basis": gri_basis,
        })

    # Sort mutations by savings
    mutation_results.sort(key=lambda m: -m["effective_savings"])

    # Compute optimized rate (best mutation)
    best_mut = mutation_results[0] if mutation_results else None
    optimized_total = best_mut["effective_duty_rate"] if best_mut else baseline_total

    return {
        "sku_id": part_id,
        "description": description,
        "origin_country": origin_country,
        "import_country": import_country,
        "current_hts_code": hts_code or None,
        "inferred_category": category.value,
        "declared_value_per_unit": declared_value_per_unit,
        "annual_volume": annual_volume,
        "base_rate": base_rate,
        "section_232_rate": section_232,
        "section_301_rate": section_301,
        "ad_duty_rate": ad_rate,
        "cvd_duty_rate": cvd_rate,
        "exclusion_relief_rate": exclusion_relief,
        "fta_preference_pct": fta_pct,
        "baseline_total_rate": baseline_total,
        "optimized_total_rate": optimized_total,
        "has_adcvd_exposure": ad_rate > 0 or cvd_rate > 0,
        "mutation_results": mutation_results,
        "adcvd_orders": adcvd_orders_for_report,
        "adcvd_confidence": adcvd_confidence,
        "fta_result": fta_result_for_report,
        "exclusion_result": excl_result_for_report,
        "requires_manual_review": False,
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _opp_to_out(o: TariffEngineeringOpportunity) -> OpportunityOut:
    return OpportunityOut(
        opportunity_id=o.opportunity_id,
        sku_id=o.sku_id,
        opportunity_type=o.opportunity_type.value,
        confidence=o.confidence.value,
        risk_grade=o.risk_grade.value,
        title=o.title,
        description=o.description,
        baseline_total_rate=o.baseline_total_rate,
        optimized_total_rate=o.optimized_total_rate,
        rate_reduction_pct=o.rate_reduction_pct,
        annual_savings_estimate=o.annual_savings_estimate,
        savings_per_unit=o.savings_per_unit,
        action_items=o.action_items,
        evidence_required=o.evidence_required,
        legal_basis=o.legal_basis,
        requires_professional_review=o.requires_professional_review,
        is_risk_finding=o.is_risk_finding,
        current_hts_code=o.current_hts_code,
        target_hts_code=o.target_hts_code,
        current_origin=o.current_origin,
    )


def _risk_to_out(r: Any) -> RiskFindingOut:
    return RiskFindingOut(
        risk_id=r.risk_id,
        sku_id=r.sku_id,
        description=r.description,
        category=r.category.value if hasattr(r.category, "value") else str(r.category),
        severity=r.severity.value if hasattr(r.severity, "value") else str(r.severity),
        estimated_exposure_pct=r.estimated_exposure_pct,
        estimated_annual_exposure_usd=r.estimated_annual_exposure_usd,
        potential_penalty_usd=r.potential_penalty_usd,
        penalty_basis=r.penalty_basis,
        risk_summary=r.risk_summary,
        risk_detail=r.risk_detail,
        immediate_actions=list(r.immediate_actions),
        remediation_steps=list(r.remediation_steps),
        legal_citations=list(r.legal_citations),
        confidence=r.confidence,
        requires_legal_counsel=r.requires_legal_counsel,
        prior_disclosure_recommended=r.prior_disclosure_recommended,
    )


def _report_to_out(report: BOMEngineeringReport) -> BOMEngineeringReportOut:
    from potatobacon.tariff.bom_engineering_report import PortfolioSummary

    ps = report.portfolio_summary
    rs = report.risk_summary

    risk_summary_out = None
    if rs is not None:
        risk_summary_out = RiskSummaryOut(
            total_risk_findings=rs.total_risk_findings,
            critical_count=rs.critical_count,
            high_count=rs.high_count,
            medium_count=rs.medium_count,
            low_count=rs.low_count,
            total_estimated_exposure_usd=rs.total_estimated_exposure_usd,
            total_potential_penalty_usd=rs.total_potential_penalty_usd,
            top_risk_categories=rs.top_risk_categories,
            prior_disclosure_recommended=rs.prior_disclosure_recommended,
            legal_counsel_required=rs.legal_counsel_required,
            overall_risk_level=rs.overall_risk_level,
            executive_summary=rs.executive_summary,
        )

    return BOMEngineeringReportOut(
        report_id=report.report_id,
        law_context=report.law_context,
        tariff_manifest_hash=report.tariff_manifest_hash,
        tenant_id=report.tenant_id,
        analyzed_at=report.analyzed_at,
        portfolio_summary=PortfolioSummaryOut(
            total_skus=ps.total_skus,
            skus_analyzed=ps.skus_analyzed,
            skus_with_opportunities=ps.skus_with_opportunities,
            skus_with_adcvd_exposure=ps.skus_with_adcvd_exposure,
            total_annual_duty_exposure=ps.total_annual_duty_exposure,
            achievable_annual_savings=ps.achievable_annual_savings,
            weighted_avg_baseline_rate=ps.weighted_avg_baseline_rate,
            weighted_avg_optimized_rate=ps.weighted_avg_optimized_rate,
            documentation_only_count=ps.documentation_only_count,
            product_engineering_count=ps.product_engineering_count,
            trade_lane_count=ps.trade_lane_count,
            reclassification_count=ps.reclassification_count,
            fta_utilization_count=ps.fta_utilization_count,
            exclusion_filing_count=ps.exclusion_filing_count,
            adcvd_exposure_count=ps.adcvd_exposure_count,
        ),
        all_opportunities=[_opp_to_out(o) for o in report.all_opportunities],
        quick_wins=[_opp_to_out(o) for o in report.quick_wins],
        sku_findings=[
            SKUExposureOut(
                sku_id=s.sku_id,
                description=s.description,
                origin_country=s.origin_country,
                current_hts_code=s.current_hts_code,
                inferred_category=s.inferred_category,
                base_rate=s.base_rate,
                section_232_rate=s.section_232_rate,
                section_301_rate=s.section_301_rate,
                ad_duty_rate=s.ad_duty_rate,
                cvd_duty_rate=s.cvd_duty_rate,
                exclusion_relief_rate=s.exclusion_relief_rate,
                fta_preference_pct=s.fta_preference_pct,
                total_effective_rate=s.total_effective_rate,
                has_trade_remedy_exposure=s.has_trade_remedy_exposure,
                has_adcvd_exposure=s.has_adcvd_exposure,
                opportunities=[_opp_to_out(o) for o in s.opportunities],
                best_savings_pct=s.best_savings_pct,
            )
            for s in report.sku_findings
        ],
        risk_findings=[_risk_to_out(r) for r in (report.risk_findings or [])],
        risk_summary=risk_summary_out,
        warnings=report.warnings,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze-sku", response_model=BOMEngineeringReportOut)
def analyze_sku(
    req: SKUAnalysisRequest,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> BOMEngineeringReportOut:
    """Engineering analysis for a single SKU.

    Returns a BOMEngineeringReport with one SKU in sku_findings,
    all discovered opportunities ranked by savings, and a portfolio_summary
    summarizing exposure for that one item.
    """
    resolved_context = req.law_context or DEFAULT_CONTEXT_ID
    atoms: List[Any] = []
    context_meta: Dict[str, Any] = {}

    if not req.law_context:
        for candidate in ("HTS_US_LIVE", "HTS_US_2025_FULL", "HTS_US_2025_TOP5", DEFAULT_CONTEXT_ID):
            try:
                atoms, context_meta = load_atoms_for_context(candidate)
                resolved_context = candidate
                break
            except (KeyError, ValueError, FileNotFoundError):
                pass

    if not atoms:
        try:
            atoms, context_meta = load_atoms_for_context(resolved_context)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=f"Tariff context unavailable: {exc}")

    sku_result = _analyze_single_sku(
        description=req.description,
        part_id=req.part_id,
        origin_country=req.origin_country,
        import_country=req.import_country or "US",
        hts_hint=req.hts_hint,
        declared_value_per_unit=req.declared_value_per_unit,
        annual_volume=req.annual_volume,
        materials_list=req.materials,
        product_category_override=req.product_category,
        atoms=atoms,
        context_meta=context_meta,
    )

    report = build_bom_engineering_report(
        report_id=str(uuid.uuid4()),
        law_context=context_meta["context_id"],
        tariff_manifest_hash=context_meta.get("manifest_hash", ""),
        tenant_id=tenant.tenant_id,
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        sku_results=[sku_result],
    )

    get_registry().increment_usage(tenant.tenant_id)
    return _report_to_out(report)


@router.post("/analyze-bom", response_model=BOMEngineeringReportOut)
async def analyze_bom(
    file: UploadFile = File(..., description="BOM file (CSV, JSON, or XLSX)"),
    import_country: str = Form(default="US"),
    default_origin_country: Optional[str] = Form(default=None),
    default_declared_value: Optional[float] = Form(default=None),
    default_annual_volume: Optional[int] = Form(default=None),
    law_context: Optional[str] = Form(default=None),
    max_mutations_per_sku: int = Form(default=5),
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> BOMEngineeringReportOut:
    """Full BOM tariff engineering analysis.

    Upload a BOM file (CSV, JSON, or XLSX) and receive a complete engineering
    report covering every line item:

    - Total annual duty exposure (if declared_value and volume are provided)
    - Achievable savings (top opportunity per SKU)
    - All opportunities ranked by annual savings
    - Quick wins (documentation-only, available immediately)
    - Per-SKU findings with full duty breakdown

    Form parameters:
      file                   — BOM file (required)
      import_country         — Importing country (default: US)
      default_origin_country — Default origin country when not specified per line
      default_declared_value — Default unit value (USD) when not in BOM
      default_annual_volume  — Default annual volume when not in BOM
      law_context            — Override tariff context (default: auto-select broadest)
      max_mutations_per_sku  — Max optimization candidates per SKU (default: 5)
    """
    raw = await file.read()
    filename = file.filename or "upload.csv"

    # Parse the BOM
    parse_result = parse_bom_file(raw, filename)
    if not parse_result.items:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No valid line items parsed from BOM. "
                f"Warnings: {parse_result.warnings}. "
                f"Skipped rows: {len(parse_result.skipped)}."
            ),
        )

    # Load tariff context
    resolved_context = law_context or DEFAULT_CONTEXT_ID
    atoms: List[Any] = []
    context_meta: Dict[str, Any] = {}

    if not law_context:
        for candidate in ("HTS_US_LIVE", "HTS_US_2025_FULL", "HTS_US_2025_TOP5", DEFAULT_CONTEXT_ID):
            try:
                atoms, context_meta = load_atoms_for_context(candidate)
                resolved_context = candidate
                break
            except (KeyError, ValueError, FileNotFoundError):
                pass

    if not atoms:
        try:
            atoms, context_meta = load_atoms_for_context(resolved_context)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=f"Tariff context unavailable: {exc}")

    # Analyze each BOM line item
    sku_results: List[Dict[str, Any]] = []
    warnings = list(parse_result.warnings)

    if parse_result.skipped:
        warnings.append(
            f"{len(parse_result.skipped)} rows skipped during parsing "
            f"(headers, empty rows, or missing descriptions)."
        )

    for item in parse_result.items:
        origin = item.origin_country or default_origin_country
        value = item.value_usd if item.value_usd is not None else default_declared_value
        volume = int(item.quantity or 1) * (default_annual_volume or 1) if item.quantity else default_annual_volume

        materials_list: List[Dict[str, str]] = []
        if item.material:
            materials_list.append({"component": "body", "material": item.material})
        else:
            for fact_key in sorted(item.extracted_facts.keys()):
                if fact_key.startswith("material_") and item.extracted_facts[fact_key]:
                    mat = fact_key.replace("material_", "")
                    materials_list.append({"component": "body", "material": mat})

        try:
            result = _analyze_single_sku(
                description=item.description,
                part_id=item.part_id,
                origin_country=origin,
                import_country=import_country,
                hts_hint=item.hts_code,
                declared_value_per_unit=value,
                annual_volume=volume,
                materials_list=materials_list if materials_list else None,
                product_category_override=None,
                atoms=atoms,
                context_meta=context_meta,
                max_mutations=max_mutations_per_sku,
            )
            sku_results.append(result)
        except Exception as exc:
            logger.warning(
                "SKU analysis failed for row %d (%s): %s",
                item.row_number, item.description[:40], exc,
            )
            warnings.append(
                f"Row {item.row_number} ({item.description[:40]}): analysis failed — {exc}"
            )

    if not sku_results:
        raise HTTPException(
            status_code=422,
            detail="All BOM line items failed analysis. Check BOM format and content.",
        )

    report = build_bom_engineering_report(
        report_id=str(uuid.uuid4()),
        law_context=context_meta["context_id"],
        tariff_manifest_hash=context_meta.get("manifest_hash", ""),
        tenant_id=tenant.tenant_id,
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        sku_results=sku_results,
        warnings=warnings,
    )

    get_registry().increment_usage(tenant.tenant_id)
    return _report_to_out(report)


# ---------------------------------------------------------------------------
# POST /v1/engineering/classify
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    """Request for HTS classification of a product description."""

    description: str = Field(..., min_length=3)
    materials: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description='Material list for GRI 3(b) essential character analysis',
    )
    top_n: int = Field(default=5, ge=1, le=20)
    chapter_filter: Optional[List[int]] = Field(
        default=None,
        description="Restrict search to specific HTS chapters",
    )

    model_config = ConfigDict(extra="forbid")


class HTSCandidateOut(BaseModel):
    hts_code: str
    heading: str
    chapter: int
    description: str
    base_duty_rate: str
    score: float
    matched_terms: List[str]
    rationale: str

    model_config = ConfigDict(extra="forbid")


class GRIReasoningOut(BaseModel):
    gri_rule: str
    rule_text: str
    applied_test: str
    result: str
    legal_citation: str

    model_config = ConfigDict(extra="forbid")


class ClassifyResponse(BaseModel):
    """HTS classification response with GRI reasoning."""

    description: str
    winning_code: str
    winning_heading: str
    winning_description: str
    winning_chapter: int
    base_duty_rate: str
    determining_rule: str
    confidence: str
    reclassification_opportunity: bool
    gri_chain: List[GRIReasoningOut]
    gri_notes: List[str]
    candidates: List[HTSCandidateOut]
    legal_basis: List[str]

    model_config = ConfigDict(extra="forbid")


@router.post("/classify", response_model=ClassifyResponse)
def classify_product(
    req: ClassifyRequest,
    api_key: str = Depends(require_api_key),
) -> ClassifyResponse:
    """Classify a product description against the HTS schedule using GRI rules.

    Returns the winning HTS heading and the full GRI reasoning chain explaining
    WHY that heading was selected over alternatives.

    This is the classification step that precedes engineering analysis.
    A tariff engineer classifies before optimizing.
    """
    candidates = search_hts_by_description(
        req.description,
        top_n=req.top_n,
        chapter_filter=req.chapter_filter,
    )

    if not candidates:
        raise HTTPException(
            status_code=422,
            detail=(
                "No HTS heading candidates found for this description. "
                "Provide a more specific product description or add material details."
            ),
        )

    gri = apply_gri(req.description, req.materials or [], candidates)

    candidates_out = [
        HTSCandidateOut(
            hts_code=c.hts_code,
            heading=c.heading,
            chapter=c.chapter,
            description=c.description,
            base_duty_rate=c.base_duty_rate,
            score=c.score,
            matched_terms=c.matched_terms,
            rationale=c.rationale,
        )
        for c in candidates
    ]

    gri_chain_out = [
        GRIReasoningOut(
            gri_rule=r.gri_rule,
            rule_text=r.rule_text,
            applied_test=r.applied_test,
            result=r.result,
            legal_citation=r.legal_citation,
        )
        for r in gri.gri_chain
    ]

    return ClassifyResponse(
        description=req.description,
        winning_code=gri.winning_code,
        winning_heading=gri.winning_heading,
        winning_description=gri.winning_description,
        winning_chapter=gri.winning_chapter,
        base_duty_rate=gri.base_duty_rate,
        determining_rule=gri.determining_rule,
        confidence=gri.confidence,
        reclassification_opportunity=gri.reclassification_opportunity,
        gri_chain=gri_chain_out,
        gri_notes=gri.notes,
        candidates=candidates_out,
        legal_basis=gri_legal_basis(gri),
    )


# ---------------------------------------------------------------------------
# POST /v1/engineering/company-profile
# ---------------------------------------------------------------------------

class CompanyProfileCapabilities(BaseModel):
    """What the engine can and cannot recommend given the company's constraints."""

    trade_lane_feasible_origins: List[str]
    trade_lane_blocked_origins: List[str]
    product_engineering_feasible: bool
    fta_programs_already_claimed: List[str]
    fta_programs_to_evaluate: List[str]
    will_surface_grade_a: bool
    will_surface_grade_b: bool
    will_surface_grade_c: bool
    audit_risk_flag: bool

    model_config = ConfigDict(extra="forbid")


class CompanyProfileResponse(BaseModel):
    profile: dict
    capabilities: CompanyProfileCapabilities
    guidance: List[str]

    model_config = ConfigDict(extra="forbid")


@router.post("/company-profile", response_model=CompanyProfileResponse)
def set_company_profile(
    profile: CompanyProfile,
    api_key: str = Depends(require_api_key),
) -> CompanyProfileResponse:
    """Accept and validate a company profile for tariff engineering analysis.

    The profile shapes which opportunities are surfaced and suppressed:
    - Fixed origins suppress trade lane recommendations for those countries
    - Active FTA programs suppress FTA utilization findings
    - Certified products suppress product engineering recommendations
    - Risk tolerance filters opportunity grade (A/B/C)

    Returns the accepted profile with a capabilities summary and
    guidance notes for the analyst.
    """
    guidance: List[str] = []

    if not profile.active_fta_programs:
        guidance.append(
            "No active FTA programs declared. The engine will surface all applicable "
            "FTA utilization opportunities. If you are already claiming FTA preferences, "
            "add them to active_fta_programs to avoid duplicate findings."
        )

    if profile.fixed_origin_countries:
        guidance.append(
            f"Fixed origins declared: {', '.join(profile.fixed_origin_countries)}. "
            "Trade lane opportunities for these origins will be suppressed."
        )

    if profile.audit_status.value in ("active", "settlement"):
        guidance.append(
            "AUDIT STATUS: company is under active CBP audit or settlement. "
            "All findings will be flagged as requiring legal counsel before acting. "
            "Priority should be compliance remediation, not optimization."
        )

    if profile.risk_tolerance.value == "low":
        guidance.append(
            "Risk tolerance LOW: only Grade A opportunities will be surfaced. "
            "This means documentation-only and well-established FTA claims only. "
            "Reclassification and trade lane opportunities are excluded."
        )

    from potatobacon.tariff.origin_rules import _FTA_PARTNERS
    applicable_ftas = [
        fta_id
        for fta_id, (_, partners) in _FTA_PARTNERS.items()
        if any(c in partners for c in profile.primary_origin_countries)
    ] if profile.primary_origin_countries else []

    unclaimed = [f for f in applicable_ftas if not profile.fta_already_claimed(f)]
    if unclaimed:
        guidance.append(
            f"Based on your origin countries, applicable FTAs include: "
            f"{', '.join(unclaimed)}. "
            "Consider adding these to fta_programs_of_interest for evaluation."
        )

    capabilities = CompanyProfileCapabilities(
        trade_lane_feasible_origins=[
            c for c in profile.primary_origin_countries
            if profile.trade_lane_feasible(c)
        ],
        trade_lane_blocked_origins=list(profile.fixed_origin_countries),
        product_engineering_feasible=profile.product_engineering_feasible(),
        fta_programs_already_claimed=list(profile.active_fta_programs),
        fta_programs_to_evaluate=list(profile.fta_programs_of_interest) or unclaimed,
        will_surface_grade_a=True,
        will_surface_grade_b=profile.risk_tolerance.value in ("moderate", "high"),
        will_surface_grade_c=profile.risk_tolerance.value == "high",
        audit_risk_flag=profile.audit_status.value in ("active", "settlement"),
    )

    return CompanyProfileResponse(
        profile=profile.model_dump(),
        capabilities=capabilities,
        guidance=guidance,
    )


# ---------------------------------------------------------------------------
# POST /v1/engineering/export/xlsx
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    """Request to generate an Excel report from a prior BOMEngineeringReport."""

    report: BOMEngineeringReportOut
    company_name: str = Field(default="Importer of Record", max_length=100)

    model_config = ConfigDict(extra="forbid")


@router.post(
    "/export/xlsx",
    response_class=Response,
    summary="Export engineering report as Excel workbook",
    description=(
        "Generate a professional Excel workbook from a BOMEngineeringReport. "
        "The workbook contains: Executive Summary, All Opportunities, Quick Wins, "
        "Per-SKU Duty Breakdown, Compliance Risk Findings, and Implementation Roadmap."
    ),
)
def export_xlsx(
    req: ExportRequest,
    api_key: str = Depends(require_api_key),
) -> Response:
    """Export a BOMEngineeringReport as a multi-sheet Excel workbook.

    The workbook is suitable for direct delivery to a client's CFO and
    trade compliance team.  It contains all findings, risk analysis,
    and a color-coded implementation roadmap.
    """
    try:
        from potatobacon.tariff.excel_generator import generate_excel_report
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Excel export not available: {exc}",
        )

    # Reconstruct a lightweight report object that excel_generator can consume.
    # We pass the BOMEngineeringReportOut directly since the generator uses duck typing.
    try:
        xlsx_bytes = generate_excel_report(
            report=_reconstitute_report_for_export(req.report),
            company_name=req.company_name,
        )
    except Exception as exc:
        logger.error("Excel generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Excel generation failed: {exc}",
        )

    filename = f"tariff_engineering_report_{req.report.report_id[:8]}.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(xlsx_bytes)),
        },
    )


class _ExportReport:
    """Lightweight shim that wraps BOMEngineeringReportOut for the Excel generator.

    The Excel generator consumes a BOMEngineeringReport (dataclass).
    This shim presents the same interface from the Pydantic API model
    without requiring full deserialization back to the dataclass.
    """

    def __init__(self, out: BOMEngineeringReportOut) -> None:
        self.report_id = out.report_id
        self.law_context = out.law_context
        self.tariff_manifest_hash = out.tariff_manifest_hash
        self.tenant_id = out.tenant_id
        self.analyzed_at = out.analyzed_at
        self.portfolio_summary = out.portfolio_summary
        self.risk_summary = out.risk_summary
        self.warnings = out.warnings

        # Wrap opportunities as simple objects for the generator
        self.all_opportunities = [_OpportunityShim(o) for o in out.all_opportunities]
        self.quick_wins = [_OpportunityShim(o) for o in out.quick_wins]
        self.sku_findings = [_SKUShim(s, out) for s in out.sku_findings]
        self.risk_findings = [_RiskShim(r) for r in out.risk_findings]


class _OpportunityShim:
    """Duck-typed shim around OpportunityOut for the Excel generator."""

    def __init__(self, o: OpportunityOut) -> None:
        from potatobacon.tariff.engineering_opportunity import (
            OpportunityConfidence, OpportunityRisk, OpportunityType
        )
        self.opportunity_id = o.opportunity_id
        self.sku_id = o.sku_id
        self.opportunity_type = o.opportunity_type          # Already a string from the model
        self.confidence = o.confidence
        self.risk_grade = o.risk_grade
        self.title = o.title
        self.description = o.description
        self.baseline_total_rate = o.baseline_total_rate
        self.optimized_total_rate = o.optimized_total_rate
        self.rate_reduction_pct = o.rate_reduction_pct
        self.annual_savings_estimate = o.annual_savings_estimate
        self.savings_per_unit = o.savings_per_unit
        self.action_items = o.action_items
        self.evidence_required = o.evidence_required
        self.legal_basis = o.legal_basis
        self.is_risk_finding = o.is_risk_finding
        self.current_hts_code = o.current_hts_code
        self.target_hts_code = o.target_hts_code
        self.current_origin = o.current_origin
        self.requires_professional_review = o.requires_professional_review
        self.estimated_implementation_cost = None
        self.implementation_time_days = None

    @property
    def payback_months(self):
        if not self.annual_savings_estimate or not self.estimated_implementation_cost:
            return None
        monthly = self.annual_savings_estimate / 12.0
        if monthly <= 0:
            return None
        return self.estimated_implementation_cost / monthly


class _SKUShim:
    """Duck-typed shim around SKUExposureOut."""

    def __init__(self, s: SKUExposureOut, report: BOMEngineeringReportOut) -> None:
        self.sku_id = s.sku_id
        self.description = s.description
        self.origin_country = s.origin_country
        self.current_hts_code = s.current_hts_code
        self.inferred_category = s.inferred_category
        self.base_rate = s.base_rate
        self.section_232_rate = s.section_232_rate
        self.section_301_rate = s.section_301_rate
        self.ad_duty_rate = s.ad_duty_rate
        self.cvd_duty_rate = s.cvd_duty_rate
        self.exclusion_relief_rate = s.exclusion_relief_rate
        self.fta_preference_pct = s.fta_preference_pct
        self.total_effective_rate = s.total_effective_rate
        self.has_trade_remedy_exposure = s.has_trade_remedy_exposure
        self.has_adcvd_exposure = s.has_adcvd_exposure
        self.best_savings_pct = s.best_savings_pct
        self.opportunities = [_OpportunityShim(o) for o in s.opportunities]


class _RiskShim:
    """Duck-typed shim around RiskFindingOut."""

    def __init__(self, r: RiskFindingOut) -> None:
        self.risk_id = r.risk_id
        self.sku_id = r.sku_id
        self.description = r.description
        self.category = r.category
        self.severity = r.severity
        self.estimated_exposure_pct = r.estimated_exposure_pct
        self.estimated_annual_exposure_usd = r.estimated_annual_exposure_usd
        self.potential_penalty_usd = r.potential_penalty_usd
        self.penalty_basis = r.penalty_basis
        self.risk_summary = r.risk_summary
        self.risk_detail = r.risk_detail
        self.immediate_actions = r.immediate_actions
        self.remediation_steps = r.remediation_steps
        self.legal_citations = r.legal_citations
        self.confidence = r.confidence
        self.requires_legal_counsel = r.requires_legal_counsel
        self.prior_disclosure_recommended = r.prior_disclosure_recommended


def _reconstitute_report_for_export(out: BOMEngineeringReportOut) -> "_ExportReport":
    """Wrap an API report model for consumption by the Excel generator."""
    return _ExportReport(out)
