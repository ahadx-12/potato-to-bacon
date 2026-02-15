from __future__ import annotations

from dataclasses import dataclass, asdict
import logging
from typing import Any, Sequence

from potatobacon.tariff.auto_classifier import classify_with_thresholds
from potatobacon.tariff.duty_calculator import DutyBreakdown, compute_total_duty
from potatobacon.tariff.reclassification_engine import find_reclassification_candidates

logger = logging.getLogger(__name__)

_ORIGIN_CANDIDATES: tuple[str, ...] = (
    "MX",
    "CA",
    "KR",
    "AU",
    "IL",
    "SG",
    "CL",
    "CO",
    "PE",
    "PA",
    "JO",
    "BH",
    "MA",
    "OM",
    "GT",
    "HN",
    "SV",
    "NI",
    "CR",
    "DO",
    "VN",
    "IN",
    "BD",
    "TH",
    "TW",
    "ID",
    "MY",
)

_HIGH_FEAS = {"MX", "CA"}
_MED_FEAS = {"KR", "VN", "TH", "IN", "MY", "TW", "ID", "SG"}


@dataclass(frozen=True)
class OptimizationStrategy:
    type: str  # origin_shift | reclassification | product_modification | first_sale | drawback
    category: str  # computed | advisory
    description: str
    estimated_savings_per_unit: float
    estimated_annual_savings: float
    risk_level: str
    confidence: float
    implementation_detail: str
    new_hts_code: str | None = None
    new_origin: str | None = None
    new_total_duty_rate: float | None = None


@dataclass(frozen=True)
class OptimizationResult:
    sku_id: str
    current_duty: DutyBreakdown
    strategies: list[OptimizationStrategy]


@dataclass(frozen=True)
class PortfolioOptimization:
    total_current_duty: float
    total_optimized_duty: float
    total_advisory_savings_potential: float
    grouped_recommendations: dict[str, dict[str, Any]]
    top_10_strategies: list[dict[str, Any]]
    risk_exposure_summary: dict[str, float]
    results: list[OptimizationResult]


def _feasibility(country: str) -> str:
    if country in _HIGH_FEAS:
        return "high"
    if country in _MED_FEAS:
        return "medium"
    return "low"


def _to_materials(materials: Sequence[str] | str) -> list[str]:
    if isinstance(materials, str):
        return [materials]
    return [str(item) for item in materials]


def optimize_sku(
    hts_code: str | None,
    description: str,
    materials: Sequence[str] | str,
    value_usd: float,
    weight_kg: float | None,
    origin_country: str,
    annual_volume: int,
    sku_id: str = "",
    intended_use: str | None = None,
) -> OptimizationResult:
    """Generate computed and advisory optimization strategies for one SKU."""

    materials_list = _to_materials(materials)
    resolved_hts = hts_code
    if not resolved_hts:
        cls = classify_with_thresholds(
            description=description,
            materials=materials_list,
            weight_kg=weight_kg,
            value_usd=value_usd,
            intended_use=intended_use,
        )
        resolved_hts = cls.get("selected_hts_code")
        if not resolved_hts:
            # Fallback to top candidate even when manual review is needed.
            candidates = cls.get("candidates", [])
            resolved_hts = candidates[0]["hts_code"] if candidates else ""
    if not resolved_hts:
        raise ValueError("Unable to resolve HTS code for optimize_sku.")

    current_duty = compute_total_duty(
        hts_code=resolved_hts,
        origin_country=origin_country,
        import_country="US",
        declared_value=value_usd,
        weight_kg=weight_kg,
    )
    baseline_rate = current_duty.total_duty_rate

    strategies: list[OptimizationStrategy] = []

    # Strategy: origin shift
    for candidate_origin in _ORIGIN_CANDIDATES:
        if candidate_origin == origin_country.upper():
            continue
        try:
            shifted = compute_total_duty(
                hts_code=resolved_hts,
                origin_country=candidate_origin,
                import_country="US",
                declared_value=value_usd,
                weight_kg=weight_kg,
            )
        except Exception:
            continue
        if shifted.total_duty_rate >= baseline_rate:
            continue
        savings_per_unit = (baseline_rate - shifted.total_duty_rate) / 100.0 * value_usd
        strategies.append(
            OptimizationStrategy(
                type="origin_shift",
                category="computed",
                description=f"Shift origin from {origin_country} to {candidate_origin} to reduce landed duty.",
                estimated_savings_per_unit=savings_per_unit,
                estimated_annual_savings=savings_per_unit * annual_volume,
                risk_level="medium",
                confidence=0.75,
                implementation_detail=f"Feasibility: {_feasibility(candidate_origin)}. Validate FTA/rules-of-origin eligibility.",
                new_origin=candidate_origin,
                new_total_duty_rate=shifted.total_duty_rate,
            )
        )

    # Strategy: reclassification
    reclass = find_reclassification_candidates(
        hts_code=resolved_hts,
        description=description,
        materials=materials_list,
        value_usd=value_usd,
        origin_country=origin_country,
    )
    for candidate in reclass[:3]:
        strategies.append(
            OptimizationStrategy(
                type="reclassification",
                category="computed",
                description=f"Reclassify from {resolved_hts} to {candidate.hts_code}: {candidate.risk_rationale}",
                estimated_savings_per_unit=candidate.savings_vs_current,
                estimated_annual_savings=candidate.savings_vs_current * annual_volume,
                risk_level=candidate.risk_level,
                confidence=candidate.plausibility_score,
                implementation_detail="Prepare classification memo and request broker/CBP validation.",
                new_hts_code=candidate.hts_code,
                new_total_duty_rate=candidate.total_duty_rate,
            )
        )

    # Strategy: product modification advisory
    cross_chapter = [item for item in reclass if item.search_type == "cross_chapter"]
    if len(materials_list) >= 2 and cross_chapter:
        best_cross = cross_chapter[0]
        if best_cross.savings_vs_current > 1.0:
            strategies.append(
                OptimizationStrategy(
                    type="product_modification",
                    category="advisory",
                    description=(
                        "Material composition appears near a chapter boundary; adjusting primary material may support "
                        f"classification under {best_cross.hts_code}."
                    ),
                    estimated_savings_per_unit=best_cross.savings_vs_current,
                    estimated_annual_savings=best_cross.savings_vs_current * annual_volume,
                    risk_level="high",
                    confidence=max(0.45, best_cross.plausibility_score),
                    implementation_detail="Requires engineering change, recertification, and classification re-validation.",
                    new_hts_code=best_cross.hts_code,
                    new_total_duty_rate=best_cross.total_duty_rate,
                )
            )

    # Strategy: first sale valuation advisory
    if origin_country.upper() in {"CN", "VN", "BD"} and value_usd > 20:
        low = baseline_rate / 100.0 * value_usd * 0.15
        high = baseline_rate / 100.0 * value_usd * 0.30
        midpoint = (low + high) / 2.0
        strategies.append(
            OptimizationStrategy(
                type="first_sale",
                category="advisory",
                description=(
                    "First sale valuation may apply if goods move through a trading company. "
                    f"Estimated savings range: ${low:.2f}-${high:.2f} per unit."
                ),
                estimated_savings_per_unit=midpoint,
                estimated_annual_savings=midpoint * annual_volume,
                risk_level="medium",
                confidence=0.6,
                implementation_detail="Requires arm's-length first-sale invoice trail and transfer-pricing support.",
            )
        )

    # Strategy: duty drawback advisory
    text_tokens = set(description.lower().split())
    if len(materials_list) >= 5 or {"component", "assembly", "sub-assembly", "module"} & text_tokens:
        annual_recovery = baseline_rate / 100.0 * value_usd * annual_volume * 0.30 * 0.99
        strategies.append(
            OptimizationStrategy(
                type="drawback",
                category="advisory",
                description=(
                    "Duty drawback may recover up to 99% of duties for exported finished goods "
                    "using imported components."
                ),
                estimated_savings_per_unit=annual_recovery / max(annual_volume, 1),
                estimated_annual_savings=annual_recovery,
                risk_level="medium",
                confidence=0.55,
                implementation_detail="Requires import/export linkage and drawback-claim controls.",
            )
        )

    strategies.sort(
        key=lambda item: (
            -item.estimated_annual_savings,
            0 if item.category == "computed" else 1,
            item.risk_level,
            item.type,
        )
    )
    return OptimizationResult(sku_id=sku_id or resolved_hts, current_duty=current_duty, strategies=strategies)


def optimize_portfolio(skus: list[dict[str, Any]]) -> PortfolioOptimization:
    results: list[OptimizationResult] = []
    total_current = 0.0
    total_optimized = 0.0
    advisory_total = 0.0

    grouped: dict[str, dict[str, Any]] = {
        key: {"count": 0, "annual_savings": 0.0}
        for key in ["origin_shift", "reclassification", "product_modification", "first_sale", "drawback"]
    }
    flat: list[dict[str, Any]] = []
    risk_summary = {
        "section_301_exposure": 0.0,
        "ad_cvd_exposure": 0.0,
        "section_232_exposure": 0.0,
    }

    for sku in skus:
        res = optimize_sku(
            hts_code=sku.get("hts_code"),
            description=sku["description"],
            materials=sku.get("materials") or sku.get("material") or "",
            value_usd=float(sku["value_usd"]),
            weight_kg=sku.get("weight_kg"),
            origin_country=sku["origin_country"],
            annual_volume=int(sku.get("annual_volume", 1)),
            sku_id=str(sku.get("part_id") or sku.get("sku_id") or ""),
            intended_use=sku.get("intended_use"),
        )
        results.append(res)

        declared_total = res.current_duty.total_duty_rate / 100.0 * float(sku["value_usd"]) * int(sku.get("annual_volume", 1))
        total_current += declared_total
        best_computed = next((s for s in res.strategies if s.category == "computed"), None)
        if best_computed and best_computed.new_total_duty_rate is not None:
            total_optimized += best_computed.new_total_duty_rate / 100.0 * float(sku["value_usd"]) * int(sku.get("annual_volume", 1))
        else:
            total_optimized += declared_total

        for strategy in res.strategies:
            grouped[strategy.type]["count"] += 1
            grouped[strategy.type]["annual_savings"] += strategy.estimated_annual_savings
            if strategy.category == "advisory":
                advisory_total += strategy.estimated_annual_savings
            flat.append(
                {
                    "sku_id": res.sku_id,
                    "type": strategy.type,
                    "category": strategy.category,
                    "description": strategy.description,
                    "estimated_annual_savings": strategy.estimated_annual_savings,
                    "risk_level": strategy.risk_level,
                }
            )

        risk_summary["section_301_exposure"] += res.current_duty.section_301_rate / 100.0 * float(sku["value_usd"]) * int(
            sku.get("annual_volume", 1)
        )
        risk_summary["ad_cvd_exposure"] += (
            (res.current_duty.ad_duty_rate + res.current_duty.cvd_duty_rate)
            / 100.0
            * float(sku["value_usd"])
            * int(sku.get("annual_volume", 1))
        )
        risk_summary["section_232_exposure"] += res.current_duty.section_232_rate / 100.0 * float(sku["value_usd"]) * int(
            sku.get("annual_volume", 1)
        )

    flat.sort(key=lambda item: -item["estimated_annual_savings"])
    top_10 = flat[:10]
    grouped_rendered = {
        key: {"count": value["count"], "annual_savings": round(value["annual_savings"], 2)}
        for key, value in grouped.items()
    }
    return PortfolioOptimization(
        total_current_duty=round(total_current, 2),
        total_optimized_duty=round(total_optimized, 2),
        total_advisory_savings_potential=round(advisory_total, 2),
        grouped_recommendations=grouped_rendered,
        top_10_strategies=top_10,
        risk_exposure_summary={key: round(value, 2) for key, value in risk_summary.items()},
        results=results,
    )


def optimization_result_to_dict(result: OptimizationResult) -> dict[str, Any]:
    return {
        "sku_id": result.sku_id,
        "current_duty": asdict(result.current_duty),
        "strategies": [asdict(item) for item in result.strategies],
    }
