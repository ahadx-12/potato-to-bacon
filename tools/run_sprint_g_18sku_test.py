#!/usr/bin/env python3
"""Sprint G: 18-SKU End-to-End Test (Auto-Resolved Rates).

Runs the full duty calculation pipeline against 12 automotive + 6 non-automotive
SKUs.  NO manual base rates are supplied — every rate is auto-resolved from the
MFN rate store.  Section 301 coverage comes from the full ingest (~1000+ headings),
not hand-seeded entries.

Scoring: pass/fail per SKU on overlay detection + total rate (within 1% tolerance).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from potatobacon.tariff.duty_calculator import compute_total_duty, DutyBreakdown
from potatobacon.tariff.fta_engine import get_fta_engine
from potatobacon.tariff.adcvd_registry import get_adcvd_registry
from potatobacon.tariff.exclusion_tracker import get_exclusion_tracker
from potatobacon.tariff.overlays import _load_overlay_rules, evaluate_overlays
from potatobacon.tariff.rate_store import get_rate_store
from potatobacon.proofs.proof_chain import ProofChain

# Clear all caches to pick up latest data
_load_overlay_rules.cache_clear()
get_fta_engine.cache_clear()
get_adcvd_registry.cache_clear()
get_exclusion_tracker.cache_clear()
get_rate_store.cache_clear()

# ------------------------------------------------------------------
# BOM Definition — NO base_rate_pct field; everything auto-resolves
# ------------------------------------------------------------------
@dataclass
class BOMItem:
    part_id: str
    description: str
    origin_country: str
    hts_code: str
    value_usd: float
    category: str  # "automotive" or "non-automotive"


BOM: list[BOMItem] = [
    # --- 12 Automotive SKUs ---
    BOMItem("AP-1001", "Stainless steel brake rotor disc ventilated 320mm",
            "CN", "8708.30.5090", 47.50, "automotive"),
    BOMItem("AP-1002", "Aluminum alloy wheel rim 18x8 inch cast",
            "CN", "8708.70.4530", 85.00, "automotive"),
    BOMItem("AP-1003", "Rubber hydraulic brake hose assembly 450mm",
            "CN", "8708.30.2190", 12.80, "automotive"),
    BOMItem("AP-1004", "LED headlamp assembly with plastic housing",
            "CN", "8512.20.2040", 135.00, "automotive"),
    BOMItem("AP-1005", "Engine timing chain kit steel roller chain",
            "DE", "8409.99.9190", 62.00, "automotive"),
    BOMItem("AP-1006", "Polyurethane front control arm bushing set",
            "KR", "8708.80.6590", 18.50, "automotive"),
    BOMItem("AP-1007", "Cast iron exhaust manifold 4-cylinder",
            "CN", "8409.91.9990", 95.00, "automotive"),
    BOMItem("AP-1008", "Windshield wiper motor 12V DC with linkage",
            "MX", "8501.31.4000", 42.00, "automotive"),
    BOMItem("AP-1009", "Cabin air filter HEPA activated carbon",
            "CN", "8421.39.8040", 8.50, "automotive"),
    BOMItem("AP-1010", "Transmission oil cooler aluminum brazed plate",
            "CN", "8708.99.8180", 58.00, "automotive"),
    BOMItem("AP-1011", "Steel wheel lug nuts M14x1.5 chrome plated set of 20",
            "CN", "7318.16.0085", 15.00, "automotive"),
    BOMItem("AP-1012", "Ignition coil pack epoxy-sealed with driver",
            "KR", "8511.30.0080", 38.00, "automotive"),

    # --- 6 Non-Automotive SKUs ---
    BOMItem("EL-2001", "Li-ion battery cell 3.7V 5000mAh prismatic",
            "CN", "8507.60.0020", 185.00, "non-automotive"),
    BOMItem("TX-3001", "Polyester fabric woven dyed 150cm width",
            "VN", "5407.61.0020", 4.50, "non-automotive"),
    BOMItem("PL-4001", "PP storage container with snap-lock lid 2L",
            "CN", "3924.10.4000", 6.00, "non-automotive"),
    BOMItem("MC-5001", "Ball screw assembly C7 grade 16mm pitch",
            "CN", "8483.40.5010", 42.00, "non-automotive"),
    BOMItem("FN-6001", "Office chair swivel adjustable height mesh back",
            "CN", "9401.30.8061", 89.00, "non-automotive"),
    BOMItem("AG-7001", "Drip irrigation emitter inline pressure compensating",
            "IL", "8424.82.0090", 0.35, "non-automotive"),
]


# ------------------------------------------------------------------
# Expected results
# ------------------------------------------------------------------
EXPECTED: dict[str, dict[str, Any]] = {
    # --- Automotive ---
    "AP-1001": {
        "desc": "CN brake rotor ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_base_approx": 2.5,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1002": {
        "desc": "CN aluminum wheel ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_base_approx": 2.5,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1003": {
        "desc": "CN brake hose ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_base_approx": 2.5,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1004": {
        "desc": "CN LED headlamp ch85 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_base_approx": 2.5,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1005": {
        "desc": "DE engine parts ch84 → base Free, no overlays = 0%",
        "expect_base_approx": 0.0,
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 0.0,
    },
    "AP-1006": {
        "desc": "KR bushing ch87 → base 2.5%, KORUS 100% pref → 0%",
        "expect_base_approx": 2.5,
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": True,
        "expected_total_approx": 0.0,
    },
    "AP-1007": {
        "desc": "CN exhaust manifold ch84 → base Free + 301-L3 25% = 25%",
        "expect_base_approx": 0.0,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 25.0,
    },
    "AP-1008": {
        "desc": "MX wiper motor ch85 → base 2.8%, USMCA 100% pref → 0%",
        "expect_base_approx": 2.8,
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": True,
        "expected_total_approx": 0.0,
    },
    "AP-1009": {
        "desc": "CN air filter 8421 → base Free + 301-L4A 7.5% = 7.5%",
        "expect_base_approx": 0.0,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 7.5,
    },
    "AP-1010": {
        "desc": "CN oil cooler ch87 → base 2.5% + 301-L3 25% = 27.5%",
        "expect_base_approx": 2.5,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 27.5,
    },
    "AP-1011": {
        "desc": "CN lug nuts ch73 → base 3.7% + 232 25% + 301-L3 25% + AD 78.21% = 131.91%",
        "expect_base_approx": 3.7,
        "expect_301": True, "expect_232": True, "expect_adcvd": True, "expect_fta": False,
        "expected_total_approx": 131.91,
    },
    "AP-1012": {
        "desc": "KR ignition coil ch85 → base 2.5%, KORUS 100% pref → 0%",
        "expect_base_approx": 2.5,
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": True,
        "expected_total_approx": 0.0,
    },

    # --- Non-Automotive ---
    "EL-2001": {
        "desc": "CN Li-ion battery ch85 → base 3.4% + 301-L3 25% = 28.4%",
        "expect_base_approx": 3.4,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 28.4,
    },
    "TX-3001": {
        "desc": "VN polyester fabric ch54 → base 14.9%, no 301/FTA = 14.9%",
        "expect_base_approx": 14.9,
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 14.9,
    },
    "PL-4001": {
        "desc": "CN PP container ch39 → base 3.4% + 301-L3 25% = 28.4%",
        "expect_base_approx": 3.4,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 28.4,
    },
    "MC-5001": {
        "desc": "CN ball screw ch84 → base 3.7% + 301-L3 25% = 28.7%",
        "expect_base_approx": 3.7,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 28.7,
    },
    "FN-6001": {
        "desc": "CN office chair ch94 → base Free + 301-L3 25% = 25%",
        "expect_base_approx": 0.0,
        "expect_301": True, "expect_232": False, "expect_adcvd": False, "expect_fta": False,
        "expected_total_approx": 25.0,
    },
    "AG-7001": {
        "desc": "IL drip irrigation ch84 → base 2.4%, Israel FTA 100% pref → 0%",
        "expect_base_approx": 2.4,
        "expect_301": False, "expect_232": False, "expect_adcvd": False, "expect_fta": True,
        "expected_total_approx": 0.0,
    },
}


# ------------------------------------------------------------------
# Test runner
# ------------------------------------------------------------------
def run_test() -> dict[str, Any]:
    # Verify rate store loaded
    store = get_rate_store()
    print(f"Rate store loaded: {store.entry_count} entries\n")

    results: dict[str, dict[str, Any]] = {}
    pass_count = 0
    auto_count = 0
    proof_chains: dict[str, dict[str, Any]] = {}

    for item in BOM:
        exp = EXPECTED[item.part_id]

        # Call compute_total_duty WITHOUT base_rate — auto-resolve everything
        duty = compute_total_duty(
            hts_code=item.hts_code,
            origin_country=item.origin_country,
            import_country="US",
        )

        # Track auto-resolution
        if duty.base_rate_source.startswith("auto:"):
            auto_count += 1

        # Build proof chain
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
        })
        chain.add_step("rate_resolution", input_data={
            "hts_code": item.hts_code,
            "base_rate_source": duty.base_rate_source,
        }, output_data={
            "base_rate": duty.base_rate,
            "base_rate_source": duty.base_rate_source,
            "base_rate_warning": duty.base_rate_warning,
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
            "best_program": (
                duty.fta_result.best_program.program_id
                if duty.fta_result and duty.fta_result.best_program else None
            ),
        })
        chain.add_step("final_duty", input_data={
            "base_rate": duty.base_rate,
            "effective_base": duty.effective_base_rate,
            "overlay_total": duty.overlay_total,
            "trade_remedy_total": duty.trade_remedy_total,
        }, output_data={
            "total_duty_rate": duty.total_duty_rate,
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

        # Check base rate
        base_delta = abs(duty.base_rate - exp["expect_base_approx"])
        base_correct = base_delta <= 0.5

        # Check total duty (within 1% tolerance)
        total_delta = abs(duty.total_duty_rate - exp["expected_total_approx"])
        rate_correct = total_delta <= 1.0

        passed = overlay_correct and rate_correct and base_correct
        if passed:
            pass_count += 1

        # AD/CVD confidence info
        adcvd_confidence = "none"
        adcvd_confidence_note = ""
        if duty.adcvd_result and hasattr(duty.adcvd_result, "confidence"):
            adcvd_confidence = duty.adcvd_result.confidence
            adcvd_confidence_note = getattr(duty.adcvd_result, "confidence_note", "")

        results[item.part_id] = {
            "description": item.description,
            "category": item.category,
            "origin": item.origin_country,
            "hts_code": item.hts_code,
            "value_usd": item.value_usd,
            "base_rate": duty.base_rate,
            "base_rate_source": duty.base_rate_source,
            "base_rate_warning": duty.base_rate_warning,
            "section_232": duty.section_232_rate,
            "section_301": duty.section_301_rate,
            "ad_duty": duty.ad_duty_rate,
            "cvd_duty": duty.cvd_duty_rate,
            "exclusion_relief": duty.exclusion_relief_rate,
            "fta_preference_pct": duty.fta_preference_pct,
            "fta_program": (
                duty.fta_result.best_program.program_id
                if duty.fta_result and duty.fta_result.best_program else None
            ),
            "effective_base": duty.effective_base_rate,
            "total_duty_rate": duty.total_duty_rate,
            "duty_cost_per_unit": round(item.value_usd * duty.total_duty_rate / 100.0, 4),
            "expected_base": exp["expect_base_approx"],
            "expected_total": exp["expected_total_approx"],
            "overlay_detection": {
                "301": {"expected": exp["expect_301"], "got": got_301},
                "232": {"expected": exp["expect_232"], "got": got_232},
                "adcvd": {"expected": exp["expect_adcvd"], "got": got_adcvd},
                "fta": {"expected": exp["expect_fta"], "got": got_fta},
            },
            "adcvd_confidence": adcvd_confidence,
            "adcvd_confidence_note": adcvd_confidence_note,
            "overlay_correct": overlay_correct,
            "base_correct": base_correct,
            "rate_correct": rate_correct,
            "passed": passed,
            "proof_chain_verified": chain_data["verified"],
            "proof_final_hash": chain_data["final_hash"],
        }

    # Deal-breaker checks
    deal_breakers = {
        "AP-1008_USMCA": results["AP-1008"]["overlay_detection"]["fta"]["got"],
        "AP-1011_ADCVD": results["AP-1011"]["overlay_detection"]["adcvd"]["got"],
        "AP-1006_KORUS": results["AP-1006"]["overlay_detection"]["fta"]["got"],
        "AP-1012_KORUS": results["AP-1012"]["overlay_detection"]["fta"]["got"],
        "AG-7001_Israel_FTA": results["AG-7001"]["overlay_detection"]["fta"]["got"],
        "AP-1009_301_L4A": results["AP-1009"]["overlay_detection"]["301"]["got"],
        "EL-2001_301_L3": results["EL-2001"]["overlay_detection"]["301"]["got"],
    }

    # Count by category
    auto_pass = sum(1 for r in results.values() if r["passed"] and r["category"] == "automotive")
    non_auto_pass = sum(1 for r in results.values() if r["passed"] and r["category"] == "non-automotive")

    portfolio = {
        "total_sku_count": len(BOM),
        "automotive_count": 12,
        "non_automotive_count": 6,
        "pass_count": pass_count,
        "fail_count": len(BOM) - pass_count,
        "automotive_pass": auto_pass,
        "non_automotive_pass": non_auto_pass,
        "auto_resolved_count": auto_count,
        "rate_store_entries": store.entry_count,
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

    print("=" * 90)
    print("SPRINT G: 18-SKU AUTO-RESOLVED DUTY TEST REPORT")
    print("=" * 90)
    print(f"\nRate Store: {portfolio['rate_store_entries']} entries loaded")
    print(f"Auto-resolved rates: {portfolio['auto_resolved_count']}/{portfolio['total_sku_count']} SKUs")

    # Per-SKU results — automotive
    print("\n" + "-" * 90)
    print("AUTOMOTIVE SKUs (12)")
    print("-" * 90)
    for pid, r in sorted(results.items()):
        if r["category"] != "automotive":
            continue
        _print_sku(pid, r)

    # Per-SKU results — non-automotive
    print("\n" + "-" * 90)
    print("NON-AUTOMOTIVE SKUs (6)")
    print("-" * 90)
    for pid, r in sorted(results.items()):
        if r["category"] != "non-automotive":
            continue
        _print_sku(pid, r)

    # Deal-breaker checks
    print("\n" + "-" * 90)
    print("DEAL-BREAKER CHECKS")
    print("-" * 90)
    for name, passed in deal_breakers.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    all_db_pass = all(deal_breakers.values())
    print(f"\n  All deal-breakers passed: {'YES' if all_db_pass else 'NO'}")

    # AD/CVD confidence report
    print("\n" + "-" * 90)
    print("AD/CVD CONFIDENCE FLAGS")
    print("-" * 90)
    for pid, r in sorted(results.items()):
        if r["adcvd_confidence"] != "none":
            print(f"  {pid}: confidence={r['adcvd_confidence']}")
            if r["adcvd_confidence_note"]:
                print(f"         {r['adcvd_confidence_note']}")

    # Proof chain spot-check
    print("\n" + "-" * 90)
    print("PROOF CHAIN INTEGRITY (spot check)")
    print("-" * 90)
    for pid in ["AP-1001", "AP-1008", "AP-1011", "EL-2001", "AG-7001"]:
        pc = data["proof_chains"][pid]
        print(f"  {pid}: steps={pc['step_count']} verified={pc['verified']} hash={pc['final_hash'][:24]}...")

    # Score
    print("\n" + "=" * 90)
    print(f"SCORE: {portfolio['pass_count']}/{portfolio['total_sku_count']}")
    print(f"  Automotive:     {portfolio['automotive_pass']}/12")
    print(f"  Non-Automotive: {portfolio['non_automotive_pass']}/6")
    print("=" * 90)

    # Verdict
    if portfolio["pass_count"] == 18 and all_db_pass:
        print("\nVERDICT: FULL PASS — All 18 SKUs calculated correctly with auto-resolved rates.")
        print("  - MFN base rates auto-resolved from rate store (no manual base_rate supplied)")
        print("  - Section 301 coverage from full ingest (~1000+ headings)")
        print("  - AD/CVD confidence flags operational")
        print("  - FTA Special-column integration working (KORUS, USMCA, Israel)")
        print("  - System works across automotive AND non-automotive product categories")
    elif portfolio["pass_count"] >= 15:
        print("\nVERDICT: MOSTLY PASSING — Minor issues on some SKUs.")
    else:
        print("\nVERDICT: NEEDS WORK — Significant failures detected.")


def _print_sku(pid: str, r: dict) -> None:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"\n[{status}] {pid}: {r['description'][:55]}")
    print(f"  Origin: {r['origin']} | HTS: {r['hts_code']} | Value: ${r['value_usd']}")
    print(f"  Base: {r['base_rate']}% (source: {r['base_rate_source']}) | Expected: {r['expected_base']}%")
    if r["base_rate_warning"]:
        print(f"  WARNING: {r['base_rate_warning']}")
    layers = []
    if r["section_232"] > 0:
        layers.append(f"232:{r['section_232']}%")
    if r["section_301"] > 0:
        layers.append(f"301:{r['section_301']}%")
    if r["ad_duty"] > 0:
        layers.append(f"AD:{r['ad_duty']}%")
    if r["cvd_duty"] > 0:
        layers.append(f"CVD:{r['cvd_duty']}%")
    if r["exclusion_relief"] > 0:
        layers.append(f"Excl:-{r['exclusion_relief']}%")
    if r["fta_preference_pct"] > 0:
        layers.append(f"FTA:{r['fta_program']}({r['fta_preference_pct']}%)")
    print(f"  Layers: {' + '.join(layers) if layers else 'none'}")
    print(f"  TOTAL: {r['total_duty_rate']}% | Expected: {r['expected_total']}% | Duty/unit: ${r['duty_cost_per_unit']}")
    if not r["overlay_correct"]:
        print(f"  ** OVERLAY MISMATCH: {r['overlay_detection']}")
    if not r["base_correct"]:
        print(f"  ** BASE RATE MISMATCH: got {r['base_rate']}% vs expected {r['expected_base']}%")
    if not r["rate_correct"]:
        print(f"  ** TOTAL RATE MISMATCH: got {r['total_duty_rate']}% vs expected {r['expected_total']}%")


if __name__ == "__main__":
    data = run_test()
    print_report(data)

    # Save full results
    output_path = REPO_ROOT / "out" / "sprint_g_18sku_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "results": data["results"],
        "portfolio": data["portfolio"],
        "deal_breakers": data["deal_breakers"],
    }
    output_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\nFull results saved to: {output_path}")
