"""Portfolio-level optimization engine for TEaaS.

Analyzes SKUs across a batch to recommend overarching strategies:
1. Origin shift analysis — savings from moving sourcing to FTA countries
2. FTA utilization report — flag unclaimed preferences
3. Risk exposure summary — AD/CVD, 301 exposure, expiring exclusions
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# FTA-eligible countries and their treaty names
FTA_COUNTRIES: Dict[str, str] = {
    "MX": "USMCA",
    "CA": "USMCA",
    "KR": "KORUS",
    "AU": "AUSFTA",
    "SG": "USSFTA",
    "IL": "US-Israel FTA",
    "CL": "USCFTA",
    "CO": "US-Colombia TPA",
    "PA": "US-Panama TPA",
    "PE": "US-Peru TPA",
    "JO": "USJFTA",
    "BH": "USBFTA",
    "OM": "US-Oman FTA",
    "MA": "USMA FTA",
    "DR": "CAFTA-DR",
    "GT": "CAFTA-DR",
    "HN": "CAFTA-DR",
    "SV": "CAFTA-DR",
    "NI": "CAFTA-DR",
    "CR": "CAFTA-DR",
}

# Countries commonly targeted by Section 301 / AD/CVD
HIGH_TARIFF_ORIGINS = {"CN", "RU", "VN"}


def origin_shift_analysis(sku_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group SKUs by origin and compute savings from shifting to FTA countries.

    Returns a list of origin shift recommendations, each with:
    - from_country: current origin
    - to_country: recommended FTA origin
    - sku_count: number of affected SKUs
    - current_total_duty: total duty at current origin
    - shifted_total_duty: estimated duty at FTA origin
    - annual_savings: estimated annual savings
    """
    by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in sku_results:
        origin = r.get("origin_country", "")
        if origin:
            by_origin[origin].append(r)

    shifts: List[Dict[str, Any]] = []

    for origin, skus in by_origin.items():
        if origin in FTA_COUNTRIES:
            continue  # Already in an FTA country

        if origin not in HIGH_TARIFF_ORIGINS:
            continue  # Focus on high-tariff origins

        # Compute current total duty
        total_current = 0.0
        for r in skus:
            bd = r.get("duty_breakdown", {})
            val = (r.get("value_usd", 0) or 0) * (r.get("annual_volume", 1) or 1)
            total_rate = bd.get("total_rate", r.get("baseline_duty_rate", 0) or 0)
            total_current += val * total_rate / 100.0

        # Estimate shifted duty for top FTA countries
        for fta_country, treaty in [("MX", "USMCA"), ("KR", "KORUS"), ("VN", "CPTPP*")]:
            if fta_country == origin:
                continue

            # Estimate: remove 301/232 and apply FTA base rate reduction
            total_shifted = 0.0
            for r in skus:
                bd = r.get("duty_breakdown", {})
                val = (r.get("value_usd", 0) or 0) * (r.get("annual_volume", 1) or 1)
                base = bd.get("base_rate", r.get("baseline_duty_rate", 0) or 0)

                # FTA countries don't get 301 tariffs, may not get 232
                shifted_rate = base  # Base MFN only
                if fta_country in FTA_COUNTRIES:
                    shifted_rate = 0.0  # Full FTA preference
                total_shifted += val * shifted_rate / 100.0

            annual_savings = total_current - total_shifted
            if annual_savings > 0:
                shifts.append({
                    "from_country": origin,
                    "to_country": fta_country,
                    "treaty": treaty,
                    "sku_count": len(skus),
                    "current_total_duty": round(total_current, 2),
                    "shifted_total_duty": round(total_shifted, 2),
                    "annual_savings": round(annual_savings, 2),
                    "sku_ids": [r.get("part_id", "") for r in skus],
                })

    # Sort by savings descending
    shifts.sort(key=lambda x: -x["annual_savings"])
    return shifts


def fta_utilization_report(sku_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag SKUs from FTA countries that might not be claiming preference.

    Returns items where:
    - Origin is an FTA country
    - FTA preference is available but may not be claimed
    - Documents the potential savings
    """
    unclaimed: List[Dict[str, Any]] = []

    for r in sku_results:
        origin = r.get("origin_country", "")
        if origin not in FTA_COUNTRIES:
            continue

        bd = r.get("duty_breakdown", {})
        fta_pref = bd.get("fta_preference_pct", 0) or 0
        base_rate = bd.get("base_rate", r.get("baseline_duty_rate", 0) or 0)
        val = (r.get("value_usd", 0) or 0)
        volume = r.get("annual_volume", 1) or 1

        # Calculate potential FTA savings
        if fta_pref > 0:
            # FTA detected — flag it as claimable
            savings = val * base_rate * fta_pref / 10000 * volume

            if savings > 0:
                unclaimed.append({
                    "part_id": r.get("part_id", "Unknown"),
                    "description": r.get("description", "")[:60],
                    "origin_country": origin,
                    "fta_treaty": FTA_COUNTRIES[origin],
                    "base_rate": base_rate,
                    "fta_preference_pct": fta_pref,
                    "potential_savings": round(savings, 2),
                    "action": f"Claim {FTA_COUNTRIES[origin]} preference — only requires certificate of origin",
                })
        elif base_rate > 0 and origin in FTA_COUNTRIES:
            # Origin is FTA-eligible but no preference was computed
            # This means they might be paying MFN unnecessarily
            potential = val * base_rate / 100 * volume
            if potential > 0:
                unclaimed.append({
                    "part_id": r.get("part_id", "Unknown"),
                    "description": r.get("description", "")[:60],
                    "origin_country": origin,
                    "fta_treaty": FTA_COUNTRIES[origin],
                    "base_rate": base_rate,
                    "fta_preference_pct": 100.0,  # Potential full elimination
                    "potential_savings": round(potential, 2),
                    "action": (
                        f"Verify {FTA_COUNTRIES[origin]} eligibility — if rules of origin are met, "
                        f"file certificate of origin for immediate savings of ${potential:,.2f}/year"
                    ),
                })

    unclaimed.sort(key=lambda x: -x["potential_savings"])
    return unclaimed


def risk_exposure_summary(sku_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute portfolio-wide risk exposure.

    Returns:
    - total_adcvd_exposure: total annual AD/CVD duty amount
    - total_301_exposure: total annual Section 301 duty amount
    - total_232_exposure: total annual Section 232 duty amount
    - low_confidence_matches: SKUs with AD/CVD broad prefix match
    - expiring_exclusions: exclusions expiring within 90 days
    - risk_items: list of individual risk flags
    """
    total_adcvd = 0.0
    total_301 = 0.0
    total_232 = 0.0
    risk_items: List[Dict[str, Any]] = []
    low_confidence: List[Dict[str, Any]] = []

    for r in sku_results:
        bd = r.get("duty_breakdown", {})
        val = (r.get("value_usd", 0) or 0) * (r.get("annual_volume", 1) or 1)

        # AD/CVD exposure
        ad = (bd.get("ad_duty_rate", 0) or 0) + (bd.get("cvd_duty_rate", 0) or 0)
        if ad > 0:
            exposure = val * ad / 100
            total_adcvd += exposure
            risk_items.append({
                "part_id": r.get("part_id", "Unknown"),
                "type": "AD/CVD",
                "rate": ad,
                "exposure": round(exposure, 2),
                "reason": f"AD/CVD exposure: {ad:.2f}% — case {bd.get('ad_case_number', 'pending')}",
            })

            # Check confidence
            confidence = bd.get("adcvd_confidence", "medium")
            if confidence == "low":
                low_confidence.append({
                    "part_id": r.get("part_id", "Unknown"),
                    "reason": "AD/CVD match is broad prefix — verify scope with trade counsel",
                })

        # 301 exposure
        s301 = bd.get("section_301_rate", 0) or 0
        if s301 > 0:
            exposure_301 = val * s301 / 100
            total_301 += exposure_301

        # 232 exposure
        s232 = bd.get("section_232_rate", 0) or 0
        if s232 > 0:
            exposure_232 = val * s232 / 100
            total_232 += exposure_232

    return {
        "total_adcvd_exposure": round(total_adcvd, 2),
        "total_301_exposure": round(total_301, 2),
        "total_232_exposure": round(total_232, 2),
        "low_confidence_matches": low_confidence,
        "expiring_exclusions": [],  # Would need date tracking
        "risk_items": risk_items,
        "skus_with_adcvd": sum(1 for r in sku_results if (r.get("duty_breakdown", {}).get("ad_duty_rate", 0) or 0) > 0),
        "skus_with_301": sum(1 for r in sku_results if (r.get("duty_breakdown", {}).get("section_301_rate", 0) or 0) > 0),
        "skus_with_232": sum(1 for r in sku_results if (r.get("duty_breakdown", {}).get("section_232_rate", 0) or 0) > 0),
    }


def run_portfolio_optimization(sku_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all portfolio-level optimizations on a set of SKU results.

    Returns a dict with:
    - origin_shifts: list of origin shift recommendations
    - fta_utilization: list of unclaimed FTA opportunities
    - risk_exposure: risk exposure summary
    """
    return {
        "origin_shifts": origin_shift_analysis(sku_results),
        "fta_utilization": fta_utilization_report(sku_results),
        "risk_exposure": risk_exposure_summary(sku_results),
        "risk_items": risk_exposure_summary(sku_results).get("risk_items", []),
    }
