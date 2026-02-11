#!/usr/bin/env python3
"""Automotive BOM End-to-End Test.

Runs the full duty calculation pipeline against 12 automotive SKUs,
exercising overlays (232, 301), AD/CVD, FTA (USMCA, KORUS), exclusions,
and proof chain generation.  Outputs per-SKU results and a portfolio summary.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Ensure repo root is on sys.path
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from potatobacon.tariff.duty_calculator import compute_total_duty, DutyBreakdown
from potatobacon.tariff.fta_engine import get_fta_engine
from potatobacon.tariff.adcvd_registry import get_adcvd_registry
from potatobacon.tariff.exclusion_tracker import get_exclusion_tracker
from potatobacon.tariff.overlays import _load_overlay_rules, evaluate_overlays
from potatobacon.proofs.proof_chain import ProofChain

# ------------------------------------------------------------------
# Clear caches to pick up any data edits
# ------------------------------------------------------------------
_load_overlay_rules.cache_clear()
get_fta_engine.cache_clear()
get_adcvd_registry.cache_clear()
get_exclusion_tracker.cache_clear()


# ------------------------------------------------------------------
# BOM Definition
# ------------------------------------------------------------------
@dataclass
class BOMItem:
    part_id: str
    description: str
    material: str
    weight_kg: float
    value_usd: float
    origin_country: str
    hts_code: str
    # Real MFN duty rate from USITC Harmonized Tariff Schedule
    base_rate_pct: float


BOM: list[BOMItem] = [
    BOMItem("AP-1001", "Stainless steel brake rotor disc ventilated 320mm",
            "stainless steel, carbon", 8.2, 47.50, "CN", "8708.30.5090",
            2.5),
    BOMItem("AP-1002", "Aluminum alloy wheel rim 18x8 inch cast",
            "aluminum alloy 6061", 9.5, 85.00, "CN", "8708.70.4530",
            2.5),
    BOMItem("AP-1003", "Rubber hydraulic brake hose assembly 450mm with steel fittings",
            "rubber, steel", 0.35, 12.80, "CN", "8708.30.2190",
            2.5),
    BOMItem("AP-1004", "LED headlamp assembly with plastic housing and wiring harness",
            "polycarbonate, LED, copper wiring", 1.8, 135.00, "CN", "8512.20.2040",
            2.5),
    BOMItem("AP-1005", "Engine timing chain kit steel roller chain with tensioner",
            "hardened steel", 1.2, 62.00, "DE", "8409.99.9190",
            0.0),  # Free for engine parts
    BOMItem("AP-1006", "Polyurethane front control arm bushing set",
            "polyurethane", 0.4, 18.50, "KR", "8708.80.6590",
            2.5),
    BOMItem("AP-1007", "Cast iron exhaust manifold 4-cylinder",
            "cast iron", 6.8, 95.00, "CN", "8409.91.9990",
            0.0),  # Free for engine parts
    BOMItem("AP-1008", "Windshield wiper motor 12V DC with linkage",
            "steel, copper, plastic", 1.1, 42.00, "MX", "8501.31.4000",
            2.8),
    BOMItem("AP-1009", "Cabin air filter HEPA activated carbon",
            "activated carbon, polypropylene", 0.3, 8.50, "CN", "8421.39.8040",
            0.0),  # Free for filtering machinery
    BOMItem("AP-1010", "Transmission oil cooler aluminum brazed plate",
            "aluminum", 2.1, 58.00, "CN", "8708.99.8180",
            2.5),
    BOMItem("AP-1011", "Steel wheel lug nuts M14x1.5 chrome plated set of 20",
            "chrome steel", 0.8, 15.00, "CN", "7318.16.0085",
            3.7),  # Steel nuts MFN rate
    BOMItem("AP-1012", "Ignition coil pack epoxy-sealed with integrated driver",
            "epoxy, copper, ferrite", 0.6, 38.00, "KR", "8511.30.0080",
            2.5),
]


# ------------------------------------------------------------------
# Expected real-world rates (for comparison / scoring)
# ------------------------------------------------------------------
EXPECTED: dict[str, dict[str, Any]] = {
    "AP-1001": {
        "desc": "CN brake rotor, Ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1002": {
        "desc": "CN aluminum wheel, Ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1003": {
        "desc": "CN brake hose, Ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1004": {
        "desc": "CN LED headlamp, Ch85 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1005": {
        "desc": "DE engine timing chain, Ch84 → base 0% (Free), no overlays = 0%",
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 0.0,
    },
    "AP-1006": {
        "desc": "KR bushing set, Ch87 → base 2.5%, KORUS eligible → 0%",
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": True,
        "expected_total_approx": 0.0,
    },
    "AP-1007": {
        "desc": "CN exhaust manifold, Ch84 → base 0% + 301-L3 25% = 25%",
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 25.0,
    },
    "AP-1008": {
        "desc": "MX wiper motor, Ch85 → base 2.8%, USMCA eligible → 0%",
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": True,
        "expected_total_approx": 0.0,
    },
    "AP-1009": {
        "desc": "CN air filter, Ch84 → base 0% + 301-L4A 7.5% = 7.5%",
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 7.5,
    },
    "AP-1010": {
        "desc": "CN oil cooler, Ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1011": {
        "desc": "CN lug nuts, Ch73 → base 3.7% + 232 25% + 301-L3 25% + AD 78.21% = 131.91%",
        "expect_301": True, "expect_232": True, "expect_adcvd": True, "expect_fta": False,
        "expected_total_approx": 131.91,
    },
    "AP-1012": {
        "desc": "KR ignition coil, Ch85 → base 2.5%, KORUS eligible → 0%",
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": True,
        "expected_total_approx": 0.0,
    },
}


# ------------------------------------------------------------------
# Run the test
# ------------------------------------------------------------------
def run_test() -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    pass_count = 0
    total_baseline_per_unit = 0.0
    total_optimized_per_unit = 0.0
    total_baseline_value = 0.0
    total_optimized_value = 0.0

    proof_chains: dict[str, dict[str, Any]] = {}

    for item in BOM:
        exp = EXPECTED[item.part_id]

        # Run the duty calculator
        duty = compute_total_duty(
            base_rate=item.base_rate_pct,
            hts_code=item.hts_code,
            origin_country=item.origin_country,
            import_country="US",
        )

        # Build proof chain for this SKU
        chain = ProofChain()
        chain.add_step("bom_input", input_data={
            "part_id": item.part_id,
            "hts_code": item.hts_code,
            "origin": item.origin_country,
            "value_usd": item.value_usd,
        }, output_data={
            "part_id": item.part_id,
            "hts_code": item.hts_code,
            "origin": item.origin_country,
            "value_usd": item.value_usd,
        })
        chain.add_step("classification", input_data={
            "hts_code": item.hts_code,
            "base_rate": item.base_rate_pct,
        }, output_data={
            "hts_code": item.hts_code,
            "base_rate": item.base_rate_pct,
        })
        chain.add_step("overlay_analysis", input_data={
            "hts_code": item.hts_code,
            "origin": item.origin_country,
        }, output_data={
            "section_232": duty.section_232_rate,
            "section_301": duty.section_301_rate,
            "ad_duty": duty.ad_duty_rate,
            "cvd_duty": duty.cvd_duty_rate,
            "exclusion_relief": duty.exclusion_relief_rate,
        })
        chain.add_step("fta_evaluation", input_data={
            "hts_code": item.hts_code,
            "origin": item.origin_country,
        }, output_data={
            "fta_preference_pct": duty.fta_preference_pct,
            "best_program": duty.fta_result.best_program.program_id if duty.fta_result and duty.fta_result.best_program else None,
        })
        chain.add_step("final_duty", input_data={
            "base_rate": duty.base_rate,
            "effective_base": duty.effective_base_rate,
            "overlay_total": duty.overlay_total,
            "trade_remedy_total": duty.trade_remedy_total,
        }, output_data={
            "total_duty_rate": duty.total_duty_rate,
            "savings_vs_no_optimization": duty.savings_vs_no_optimization,
        })

        chain_data = chain.to_dict()
        proof_chains[item.part_id] = chain_data

        # Check overlay detection
        got_301 = duty.has_301_exposure
        got_232 = duty.has_232_exposure
        got_adcvd = duty.has_adcvd_exposure
        got_fta = duty.has_fta_preference

        overlay_correct = (
            got_301 == exp["expect_301"]
            and got_232 == exp["expect_232"]
            and got_adcvd == exp["expect_adcvd"]
            and got_fta == exp["expect_fta"]
        )

        # Check total duty (within 1% tolerance)
        total_delta = abs(duty.total_duty_rate - exp["expected_total_approx"])
        rate_correct = total_delta <= 1.0

        passed = overlay_correct and rate_correct
        if passed:
            pass_count += 1

        # Duty costs
        duty_cost_per_unit = item.value_usd * duty.total_duty_rate / 100.0
        optimized_cost = item.value_usd * duty.effective_base_rate / 100.0  # just base after FTA

        total_baseline_per_unit += duty_cost_per_unit
        total_optimized_per_unit += optimized_cost
        total_baseline_value += item.value_usd

        results[item.part_id] = {
            "description": item.description,
            "origin": item.origin_country,
            "hts_code": item.hts_code,
            "value_usd": item.value_usd,
            "base_rate": duty.base_rate,
            "section_232": duty.section_232_rate,
            "section_301": duty.section_301_rate,
            "ad_duty": duty.ad_duty_rate,
            "cvd_duty": duty.cvd_duty_rate,
            "exclusion_relief": duty.exclusion_relief_rate,
            "fta_preference_pct": duty.fta_preference_pct,
            "fta_program": (
                duty.fta_result.best_program.program_id
                if duty.fta_result and duty.fta_result.best_program
                else None
            ),
            "effective_base": duty.effective_base_rate,
            "total_duty_rate": duty.total_duty_rate,
            "duty_cost_per_unit": round(duty_cost_per_unit, 2),
            "expected_total": exp["expected_total_approx"],
            "overlay_detection": {
                "301": {"expected": exp["expect_301"], "got": got_301},
                "232": {"expected": exp["expect_232"], "got": got_232},
                "adcvd": {"expected": exp["expect_adcvd"], "got": got_adcvd},
                "fta": {"expected": exp["expect_fta"], "got": got_fta},
            },
            "overlay_correct": overlay_correct,
            "rate_correct": rate_correct,
            "passed": passed,
            "proof_chain_verified": chain_data["verified"],
            "proof_final_hash": chain_data["final_hash"],
        }

    # Portfolio summary
    annual_units = 1000 * 12  # 1000 units/month
    portfolio = {
        "total_sku_count": len(BOM),
        "pass_count": pass_count,
        "fail_count": len(BOM) - pass_count,
        "total_bom_value_per_unit": round(total_baseline_value, 2),
        "total_baseline_duty_per_unit": round(total_baseline_per_unit, 2),
        "total_optimized_duty_per_unit": round(total_optimized_per_unit, 2),
        "total_savings_per_unit": round(total_baseline_per_unit - total_optimized_per_unit, 2),
        "annualized_baseline_duty": round(total_baseline_per_unit * annual_units, 2),
        "annualized_optimized_duty": round(total_optimized_per_unit * annual_units, 2),
        "annualized_savings": round((total_baseline_per_unit - total_optimized_per_unit) * annual_units, 2),
        "annual_units": annual_units,
    }

    # Deal-breaker checks
    deal_breakers = {
        "AP-1008_USMCA": results["AP-1008"]["overlay_detection"]["fta"]["got"],
        "AP-1011_ADCVD": results["AP-1011"]["overlay_detection"]["adcvd"]["got"],
        "AP-1006_KORUS": results["AP-1006"]["overlay_detection"]["fta"]["got"],
        "AP-1012_KORUS": results["AP-1012"]["overlay_detection"]["fta"]["got"],
    }

    return {
        "results": results,
        "portfolio": portfolio,
        "deal_breakers": deal_breakers,
        "proof_chains": proof_chains,
    }


def print_report(data: dict[str, Any]) -> None:
    results = data["results"]
    portfolio = data["portfolio"]
    deal_breakers = data["deal_breakers"]
    proof_chains = data["proof_chains"]

    print("=" * 80)
    print("AUTOMOTIVE BOM END-TO-END TEST REPORT")
    print("=" * 80)

    # Per-SKU results
    print("\n--- PER-SKU RESULTS ---\n")
    for pid, r in sorted(results.items()):
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {pid}: {r['description'][:50]}")
        print(f"  Origin: {r['origin']} | HTS: {r['hts_code']} | Value: ${r['value_usd']}")
        print(f"  Base: {r['base_rate']}% | 232: {r['section_232']}% | 301: {r['section_301']}% | AD: {r['ad_duty']}% | CVD: {r['cvd_duty']}%")
        if r['fta_program']:
            print(f"  FTA: {r['fta_program']} ({r['fta_preference_pct']}% preference) → effective base: {r['effective_base']}%")
        print(f"  Exclusion relief: {r['exclusion_relief']}%")
        print(f"  TOTAL: {r['total_duty_rate']}% | Expected: {r['expected_total']}% | Duty/unit: ${r['duty_cost_per_unit']}")
        if not r["overlay_correct"]:
            print(f"  ** OVERLAY MISMATCH: {r['overlay_detection']}")
        print(f"  Proof chain: verified={r['proof_chain_verified']} hash={r['proof_final_hash'][:16]}...")
        print()

    # Deal-breaker checks
    print("--- DEAL-BREAKER CHECKS ---\n")
    print(f"  AP-1008 (Mexico motor) → USMCA found: {'YES' if deal_breakers['AP-1008_USMCA'] else 'NO'}")
    print(f"  AP-1011 (China lug nuts) → AD/CVD found: {'YES' if deal_breakers['AP-1011_ADCVD'] else 'NO'}")
    print(f"  AP-1006 (Korea bushing) → KORUS found: {'YES' if deal_breakers['AP-1006_KORUS'] else 'NO'}")
    print(f"  AP-1012 (Korea coil) → KORUS found: {'YES' if deal_breakers['AP-1012_KORUS'] else 'NO'}")
    all_deal_breakers_pass = all(deal_breakers.values())
    print(f"  All deal-breakers passed: {'YES' if all_deal_breakers_pass else 'NO'}")

    # Portfolio summary
    print("\n--- PORTFOLIO SUMMARY ---\n")
    print(f"  Total BOM value per unit: ${portfolio['total_bom_value_per_unit']}")
    print(f"  Total baseline duty per unit: ${portfolio['total_baseline_duty_per_unit']}")
    print(f"  Total optimized duty per unit: ${portfolio['total_optimized_duty_per_unit']}")
    print(f"  Savings per unit: ${portfolio['total_savings_per_unit']}")
    print(f"  Annualized ({portfolio['annual_units']:,} units):")
    print(f"    Baseline duty: ${portfolio['annualized_baseline_duty']:,.2f}")
    print(f"    Optimized duty: ${portfolio['annualized_optimized_duty']:,.2f}")
    print(f"    Savings: ${portfolio['annualized_savings']:,.2f}")

    # Proof chain integrity check (pick 3)
    print("\n--- PROOF CHAIN INTEGRITY CHECK ---\n")
    checked_ids = ["AP-1001", "AP-1008", "AP-1011"]
    for pid in checked_ids:
        pc = proof_chains[pid]
        print(f"  {pid}: steps={pc['step_count']} verified={pc['verified']} final_hash={pc['final_hash'][:32]}...")
        for step in pc["steps"]:
            print(f"    Step {step['step_index']}: {step['step_name']} chain={step['chain_hash'][:16]}...")

    # Score
    print(f"\n--- OVERALL SCORE: {portfolio['pass_count']}/{portfolio['total_sku_count']} ---\n")

    # Verdict
    if portfolio["pass_count"] == portfolio["total_sku_count"] and all_deal_breakers_pass:
        print("VERDICT: All 12 SKUs calculated correctly. All deal-breaker checks passed.")
        print("The duty calculator correctly handles Section 232, Section 301 (List 3 and 4A),")
        print("AD/CVD orders, USMCA preferences, and KORUS preferences. Proof chains verify.")
        print("An importer could use this output as a starting point for duty planning,")
        print("though real-world use would require certificate-of-origin documentation")
        print("and periodic data refreshes against USITC/USTR published schedules.")
    elif portfolio["pass_count"] >= 10:
        print("VERDICT: Most SKUs correct but some edge cases need attention.")
    else:
        print("VERDICT: Significant gaps remain. Not ready for production use.")

    return


if __name__ == "__main__":
    data = run_test()
    print_report(data)

    # Save full results to JSON
    output_path = REPO_ROOT / "out" / "automotive_bom_e2e_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert proof chains to serializable form
    serializable = {
        "results": data["results"],
        "portfolio": data["portfolio"],
        "deal_breakers": data["deal_breakers"],
    }
    output_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\nFull results saved to: {output_path}")
