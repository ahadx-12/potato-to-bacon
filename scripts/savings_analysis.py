#!/usr/bin/env python3
"""
Standalone TEaaS Savings Analysis — Real-World Tariff Calculator
===============================================================
Reads the 18-SKU BOM and ALL tariff data files (HTS rates, Section 301,
Section 232, AD/CVD, FTA preferences, exclusions) to compute:
  1. Baseline total landed duty per SKU
  2. Optimized duty via sourcing shifts and FTA utilization
  3. Per-SKU and portfolio-level savings
"""

import csv
import json
import os
import re
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BOM_PATH = ROOT / "test_data" / "automotive_bom_smoke_test.csv"

# ─── Load data files ───────────────────────────────────────────────
def load_json(relpath):
    with open(DATA / relpath, encoding="utf-8") as f:
        return json.load(f)

hts_seed = load_json("hts_extract/hts_rates_seed.json")
sec301 = load_json("overlays/section301_sample.json")
sec232 = load_json("overlays/section232_sample.json")
adcvd = load_json("overlays/adcvd_orders.json")
fta_data = load_json("overlays/fta_preferences.json")
exclusions_data = load_json("overlays/exclusions.json")

# ─── Build lookup tables ───────────────────────────────────────────
def parse_rate(rate_str):
    """Parse HTS rate string to a float percentage. Returns 0.0 for Free or non-standard."""
    if not rate_str:
        return 0.0
    rate_str = rate_str.strip()
    if rate_str.lower() == "free":
        return 0.0
    m = re.match(r"([\d.]+)%", rate_str)
    if m:
        return float(m.group(1))
    return 0.0  # Specific rates like ¢/kg — treat as 0 for ad-valorem analysis

# HTS code → MFN rate + special rates
hts_rates = {}
for entry in hts_seed["rates"]:
    code = entry["hts_code"]
    hts_rates[code] = {
        "mfn_rate": parse_rate(entry["general"]),
        "description": entry.get("description", ""),
        "special_rates": entry.get("special_rates", {}),
    }

# Section 301: origin + HTS prefix → additional rate
sec301_entries = []
for entry in sec301:
    sec301_entries.append({
        "name": entry["overlay_name"],
        "prefixes": set(entry["hts_prefixes"]),
        "countries": set(entry["origin_countries"]),
        "rate": entry["additional_rate"],
    })

# Section 232: HTS prefix → additional rate
sec232_entries = []
for entry in sec232:
    sec232_entries.append({
        "name": entry["overlay_name"],
        "prefixes": set(entry["hts_prefixes"]),
        "rate": entry["additional_rate"],
    })

# AD/CVD: HTS prefix + origin → duty rate
adcvd_orders = []
for order in adcvd["orders"]:
    if order["status"] == "active":
        adcvd_orders.append({
            "order_id": order["order_id"],
            "type": order["type"],
            "product": order["product_description"],
            "prefixes": order["hts_prefixes"],
            "countries": set(order["origin_countries"]),
            "rate": order["duty_rate_pct"],
        })

# Exclusions
active_exclusions = []
today = date.today()
for exc in exclusions_data["exclusions"]:
    if exc["status"] == "active":
        active_exclusions.append({
            "id": exc["exclusion_id"],
            "type": exc["overlay_type"],
            "hts_codes": exc["hts_codes"],
            "countries": set(exc.get("origin_countries", [])),
            "relief_pct": exc["exclusion_rate_pct"],
            "expiry": exc["expiry_date"],
        })

# FTA programs
fta_programs = []
for prog in fta_data["programs"]:
    if prog["status"] == "active":
        fta_programs.append({
            "id": prog["program_id"],
            "name": prog["name"],
            "countries": set(prog["partner_countries"]),
            "preference_pct": prog["default_preference_pct"],
            "rvc_threshold": prog.get("rvc_threshold_pct", 0),
        })


# ─── Tariff calculation engine ─────────────────────────────────────
def hts_prefix_match(hts_code, prefixes):
    """Check if an HTS code starts with any of the given prefixes."""
    for p in prefixes:
        if hts_code.startswith(p):
            return True
    return False

def lookup_mfn_rate(hts_code):
    """Look up MFN rate, trying exact match first then prefix matching."""
    if hts_code in hts_rates:
        return hts_rates[hts_code]["mfn_rate"]
    # Try increasingly shorter prefixes
    for length in range(len(hts_code) - 1, 3, -1):
        prefix = hts_code[:length]
        if prefix in hts_rates:
            return hts_rates[prefix]["mfn_rate"]
    # Try chapter-level (first 4 digits)
    ch = hts_code[:4]
    if ch in hts_rates:
        return hts_rates[ch]["mfn_rate"]
    return 0.0

def get_special_rate(hts_code, country):
    """Get FTA/special rate for a country from HTS data."""
    if hts_code in hts_rates:
        sr = hts_rates[hts_code].get("special_rates", {})
        return sr.get(country, None)
    return None

def compute_section301(hts_code, origin):
    """Compute Section 301 additional duty rate."""
    for entry in sec301_entries:
        if origin in entry["countries"] and hts_prefix_match(hts_code, entry["prefixes"]):
            return entry["rate"], entry["name"]
    return 0.0, None

def compute_section232(hts_code):
    """Compute Section 232 additional duty rate."""
    for entry in sec232_entries:
        if hts_prefix_match(hts_code, entry["prefixes"]):
            return entry["rate"], entry["name"]
    return 0.0, None

def compute_adcvd(hts_code, origin):
    """Compute AD/CVD duties. Returns (ad_rate, cvd_rate, details)."""
    ad_total = 0.0
    cvd_total = 0.0
    details = []
    for order in adcvd_orders:
        if origin in order["countries"] and hts_prefix_match(hts_code, order["prefixes"]):
            if order["type"] == "AD":
                ad_total += order["rate"]
            else:
                cvd_total += order["rate"]
            details.append(f"{order['order_id']} ({order['type']} {order['rate']}%): {order['product']}")
    return ad_total, cvd_total, details

def check_exclusions(hts_code, origin, overlay_type):
    """Check if any exclusion applies."""
    for exc in active_exclusions:
        if exc["type"] != overlay_type:
            continue
        for exc_code in exc["hts_codes"]:
            if hts_code.startswith(exc_code):
                if not exc["countries"] or origin in exc["countries"]:
                    return exc["relief_pct"], exc["id"]
    return 0.0, None

def find_fta_eligible_sources(hts_code, current_origin):
    """Find FTA-eligible alternative sourcing countries with Free rates."""
    alternatives = []
    for prog in fta_programs:
        for country in prog["countries"]:
            if country == current_origin:
                continue
            special = get_special_rate(hts_code, country)
            if special and special.lower() == "free":
                alternatives.append({
                    "country": country,
                    "program": prog["id"],
                    "program_name": prog["name"],
                    "rvc_threshold": prog["rvc_threshold"],
                })
    return alternatives


def compute_sku_duty(sku):
    """Compute full duty breakdown for a single SKU."""
    hts = sku["hts_code"]
    origin = sku["origin_country"]
    value = float(sku["value_usd"])
    
    # 1. Base MFN rate
    mfn = lookup_mfn_rate(hts)
    
    # 2. Section 301
    s301_rate, s301_name = compute_section301(hts, origin)
    
    # 3. Section 232
    s232_rate, s232_name = compute_section232(hts)
    
    # 4. AD/CVD
    ad_rate, cvd_rate, adcvd_details = compute_adcvd(hts, origin)
    
    # 5. Exclusion relief
    s232_exclusion, s232_exc_id = check_exclusions(hts, origin, "section_232")
    s301_exclusion, s301_exc_id = check_exclusions(hts, origin, "section_301")
    
    # Net overlay rates after exclusions
    net_s232 = max(0, s232_rate - s232_exclusion)
    net_s301 = max(0, s301_rate - s301_exclusion)
    
    # Total effective rate
    total_rate = mfn + net_s301 + net_s232 + ad_rate + cvd_rate
    duty_amount = value * (total_rate / 100.0)
    
    # FTA alternative sourcing analysis
    fta_alternatives = find_fta_eligible_sources(hts, origin)
    
    # Best optimization: shift to FTA country (eliminates MFN + avoids 301)
    optimized_rate = total_rate
    optimization = None
    
    if fta_alternatives and (net_s301 > 0 or mfn > 0):
        # Pick best FTA country (USMCA first, then KORUS)
        preferred_order = ["MX", "CA", "KR", "AU", "SG", "CL", "CO"]
        best_alt = None
        for pref in preferred_order:
            for alt in fta_alternatives:
                if alt["country"] == pref:
                    best_alt = alt
                    break
            if best_alt:
                break
        if not best_alt:
            best_alt = fta_alternatives[0]
        
        # With FTA: MFN → Free, Section 301 goes away (not CN anymore)
        # BUT Section 232 still applies if steel/aluminum (it's country-neutral)
        opt_mfn = 0.0  # FTA rate = Free
        opt_301 = 0.0  # No longer CN origin
        # Section 232 still applies to steel/aluminum regardless of origin
        # UNLESS source country has 232 exemption (CA/MX are exempt under USMCA)
        opt_232 = net_s232
        if best_alt["country"] in ("CA", "MX") and best_alt["program"] == "USMCA":
            opt_232 = 0.0  # USMCA countries exempt from 232
        
        opt_ad = 0.0  # AD/CVD is origin-specific, wouldn't apply
        opt_cvd = 0.0
        
        optimized_rate = opt_mfn + opt_301 + opt_232 + opt_ad + opt_cvd
        savings_rate = total_rate - optimized_rate
        
        if savings_rate > 0.5:  # Only report meaningful savings
            optimization = {
                "strategy": f"Source from {best_alt['country']} under {best_alt['program']}",
                "detail": f"Shift sourcing from {origin} to {best_alt['country']} "
                          f"(RVC threshold: {best_alt['rvc_threshold']}%)",
                "baseline_rate": total_rate,
                "optimized_rate": optimized_rate,
                "savings_pct": savings_rate,
                "savings_usd": value * (savings_rate / 100.0),
            }
    
    return {
        "part_id": sku["part_id"],
        "description": sku["description"],
        "hts_code": hts,
        "origin": origin,
        "value_usd": value,
        "mfn_rate": mfn,
        "s301_rate": net_s301,
        "s301_name": s301_name,
        "s232_rate": net_s232,
        "s232_name": s232_name,
        "ad_rate": ad_rate,
        "cvd_rate": cvd_rate,
        "adcvd_details": adcvd_details,
        "total_baseline_rate": total_rate,
        "baseline_duty_usd": duty_amount,
        "optimized_rate": optimized_rate,
        "optimized_duty_usd": value * (optimized_rate / 100.0),
        "optimization": optimization,
        "fta_alternatives_count": len(fta_alternatives),
    }


# ─── Main analysis ─────────────────────────────────────────────────
def main():
    # Read BOM
    skus = []
    with open(BOM_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("part_id"):
                skus.append(row)
    
    print("=" * 80)
    print("  TEaaS TARIFF SAVINGS ANALYSIS — REAL-WORLD DUTY CALCULATIONS")
    print("  BOM: automotive_bom_smoke_test.csv (18 SKUs)")
    print(f"  Date: {date.today().isoformat()}")
    print("=" * 80)
    
    results = []
    for sku in skus:
        result = compute_sku_duty(sku)
        results.append(result)
    
    # ─── Per-SKU Detail Report ──────────────────────────────────────
    total_baseline_duty = 0.0
    total_optimized_duty = 0.0
    total_value = 0.0
    savings_found = []
    high_risk_skus = []
    
    print("\n" + "─" * 80)
    print("  PER-SKU DUTY BREAKDOWN")
    print("─" * 80)
    
    for r in results:
        total_baseline_duty += r["baseline_duty_usd"]
        total_optimized_duty += r["optimized_duty_usd"]
        total_value += r["value_usd"]
        
        has_overlay = r["s301_rate"] > 0 or r["s232_rate"] > 0 or r["ad_rate"] > 0 or r["cvd_rate"] > 0
        
        print(f"\n{'━' * 78}")
        status = "⚠️  HIGH EXPOSURE" if r["total_baseline_rate"] > 30 else ("📋 DUTY-FREE" if r["total_baseline_rate"] == 0 else "📦 STANDARD")
        print(f"  {r['part_id']} | {r['description'][:50]}")
        print(f"  HTS: {r['hts_code']}  |  Origin: {r['origin']}  |  Value: ${r['value_usd']:.2f}  |  {status}")
        print(f"  {'─' * 74}")
        
        # Duty layers
        print(f"  Base MFN Rate:          {r['mfn_rate']:6.1f}%     ${r['value_usd'] * r['mfn_rate'] / 100:.2f}")
        if r["s301_rate"] > 0:
            print(f"  + Section 301:          {r['s301_rate']:6.1f}%     ${r['value_usd'] * r['s301_rate'] / 100:.2f}  ({r['s301_name']})")
        if r["s232_rate"] > 0:
            print(f"  + Section 232:          {r['s232_rate']:6.1f}%     ${r['value_usd'] * r['s232_rate'] / 100:.2f}  ({r['s232_name']})")
        if r["ad_rate"] > 0:
            print(f"  + Antidumping:          {r['ad_rate']:6.1f}%     ${r['value_usd'] * r['ad_rate'] / 100:.2f}")
        if r["cvd_rate"] > 0:
            print(f"  + Countervailing:       {r['cvd_rate']:6.1f}%     ${r['value_usd'] * r['cvd_rate'] / 100:.2f}")
        
        print(f"  {'─' * 74}")
        print(f"  TOTAL LANDED DUTY:      {r['total_baseline_rate']:6.1f}%     ${r['baseline_duty_usd']:.2f}")
        
        if r["optimization"]:
            opt = r["optimization"]
            print(f"\n  ✅ OPTIMIZATION: {opt['strategy']}")
            print(f"     {opt['detail']}")
            print(f"     Baseline: {opt['baseline_rate']:.1f}% → Optimized: {opt['optimized_rate']:.1f}% = SAVES {opt['savings_pct']:.1f}% (${opt['savings_usd']:.2f}/unit)")
            savings_found.append(r)
        
        if r["total_baseline_rate"] > 30:
            high_risk_skus.append(r)
        
        if r["adcvd_details"]:
            print(f"\n  ⚡ AD/CVD ORDERS:")
            for d in r["adcvd_details"]:
                print(f"     • {d}")
    
    # ─── Portfolio Summary ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  PORTFOLIO SUMMARY")
    print("=" * 80)
    
    total_savings_usd = total_baseline_duty - total_optimized_duty
    
    print(f"\n  Total Portfolio Value:        ${total_value:>12,.2f}")
    print(f"  Total Baseline Duty:         ${total_baseline_duty:>12,.2f}  ({total_baseline_duty/total_value*100:.1f}% effective rate)")
    print(f"  Total Optimized Duty:        ${total_optimized_duty:>12,.2f}  ({total_optimized_duty/total_value*100:.1f}% effective rate)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  TOTAL SAVINGS:               ${total_savings_usd:>12,.2f}  ({total_savings_usd/total_value*100:.1f}% of value)")
    
    # Annualize (assume 1000 units/year per SKU)
    annual_volume = 1000
    print(f"\n  Annualized @ {annual_volume} units/SKU/year:")
    print(f"    Annual duty without optimization: ${total_baseline_duty * annual_volume:>14,.2f}")
    print(f"    Annual duty WITH optimization:    ${total_optimized_duty * annual_volume:>14,.2f}")
    print(f"    ANNUAL SAVINGS:                   ${total_savings_usd * annual_volume:>14,.2f}")
    
    # ─── Savings Breakdown ──────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  ACTIONABLE SAVINGS OPPORTUNITIES (Top → Bottom by Impact)")
    print("=" * 80)
    
    savings_found.sort(key=lambda x: x["optimization"]["savings_usd"], reverse=True)
    
    for i, r in enumerate(savings_found, 1):
        opt = r["optimization"]
        print(f"\n  {i}. {r['part_id']} — {r['description'][:45]}")
        print(f"     Strategy:   {opt['strategy']}")
        print(f"     Savings:    {opt['savings_pct']:.1f}% = ${opt['savings_usd']:.2f}/unit")
        print(f"     Annual:     ${opt['savings_usd'] * annual_volume:>10,.2f}/year")
        print(f"     Complexity: {'Low' if opt['optimized_rate'] == 0 else 'Medium'} — {opt['detail']}")
    
    # ─── Risk Exposure ──────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  RISK EXPOSURE SUMMARY")
    print("=" * 80)
    
    # Section 301 exposure
    s301_skus = [r for r in results if r["s301_rate"] > 0]
    s301_total = sum(r["value_usd"] * r["s301_rate"] / 100 for r in s301_skus)
    print(f"\n  Section 301 (PRC Tariffs):")
    print(f"    Affected SKUs: {len(s301_skus)}")
    print(f"    Total 301 duty: ${s301_total:,.2f}/batch (${s301_total * annual_volume:,.2f}/year)")
    for r in s301_skus:
        print(f"      • {r['part_id']} ({r['hts_code']}): +{r['s301_rate']:.0f}% = ${r['value_usd'] * r['s301_rate'] / 100:.2f}")
    
    # Section 232 exposure
    s232_skus = [r for r in results if r["s232_rate"] > 0]
    s232_total = sum(r["value_usd"] * r["s232_rate"] / 100 for r in s232_skus)
    print(f"\n  Section 232 (Steel/Aluminum Safeguard):")
    print(f"    Affected SKUs: {len(s232_skus)}")
    print(f"    Total 232 duty: ${s232_total:,.2f}/batch (${s232_total * annual_volume:,.2f}/year)")
    for r in s232_skus:
        print(f"      • {r['part_id']} ({r['hts_code']}): +{r['s232_rate']:.0f}% = ${r['value_usd'] * r['s232_rate'] / 100:.2f}")
    
    # AD/CVD exposure
    adcvd_skus = [r for r in results if r["ad_rate"] > 0 or r["cvd_rate"] > 0]
    adcvd_total = sum(r["value_usd"] * (r["ad_rate"] + r["cvd_rate"]) / 100 for r in adcvd_skus)
    print(f"\n  AD/CVD (Antidumping/Countervailing):")
    print(f"    Affected SKUs: {len(adcvd_skus)}")
    print(f"    Total AD/CVD duty: ${adcvd_total:,.2f}/batch (${adcvd_total * annual_volume:,.2f}/year)")
    for r in adcvd_skus:
        print(f"      • {r['part_id']} ({r['hts_code']}): AD {r['ad_rate']:.1f}%, CVD {r['cvd_rate']:.1f}% = ${r['value_usd'] * (r['ad_rate'] + r['cvd_rate']) / 100:.2f}")
    
    # Duty-free SKUs
    free_skus = [r for r in results if r["total_baseline_rate"] == 0]
    print(f"\n  Duty-Free SKUs: {len(free_skus)}")
    for r in free_skus:
        print(f"    • {r['part_id']} ({r['hts_code']}, {r['origin']}): MFN Free")
    
    print(f"\n{'=' * 80}")
    print("  END OF ANALYSIS")
    print("=" * 80)

if __name__ == "__main__":
    main()
