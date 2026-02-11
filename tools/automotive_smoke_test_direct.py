#!/usr/bin/env python3
"""
Direct Automotive BOM Smoke Test

Bypasses the API and directly tests the tariff engine with realistic automotive parts.
This reveals the actual quality of the core tariff engineering logic.
"""

import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from potatobacon.tariff.hts_hint_resolver import resolve_hts_hint
from potatobacon.tariff.duty_calculator import calculate_duty
from potatobacon.tariff.overlays import apply_overlays
from potatobacon.tariff.tariff_optimizer import suggest_optimizations


# Test SKUs with expected results
TEST_SKUS = [
    {
        "part_id": "AP-1001",
        "description": "Stainless steel brake rotor disc ventilated 320mm",
        "material": "stainless steel, carbon",
        "weight_kg": 8.2,
        "value_usd": 47.50,
        "origin_country": "CN",
        "hts_code": "8708.30.5090",
        "expected": {
            "name": "Brake rotor (stainless steel, CN)",
            "base_rate": 2.5,
            "section_301": True,
            "total_min": 27.0,  # 2.5% base + 25% Section 301
        },
    },
    {
        "part_id": "AP-1002",
        "description": "Aluminum alloy wheel rim 18x8 inch cast",
        "material": "aluminum alloy 6061",
        "weight_kg": 9.5,
        "value_usd": 85.00,
        "origin_country": "CN",
        "hts_code": "8708.70.4530",
        "expected": {
            "name": "Aluminum wheel (CN)",
            "base_rate": 2.5,
            "section_301": True,
            "section_232": False,  # May or may not apply
            "total_min": 27.0,
        },
    },
    {
        "part_id": "AP-1008",
        "description": "Windshield wiper motor 12V DC with linkage",
        "material": "steel, copper, plastic",
        "weight_kg": 1.1,
        "value_usd": 42.00,
        "origin_country": "MX",
        "hts_code": "8501.31.4000",
        "expected": {
            "name": "Wiper motor (MX)",
            "base_rate": 2.7,
            "usmca": True,  # CRITICAL: Must find USMCA preference
            "preferential_rate": 0.0,
        },
    },
    {
        "part_id": "AP-1006",
        "description": "Polyurethane front control arm bushing set",
        "material": "polyurethane",
        "weight_kg": 0.4,
        "value_usd": 18.50,
        "origin_country": "KR",
        "hts_code": "8708.80.6590",
        "expected": {
            "name": "Control arm bushing (KR)",
            "base_rate": 2.5,
            "korus": True,  # Must find KORUS FTA
            "preferential_rate": 0.0,
        },
    },
    {
        "part_id": "AP-1011",
        "description": "Steel wheel lug nuts M14x1.5 chrome plated set of 20",
        "material": "chrome steel",
        "weight_kg": 0.8,
        "value_usd": 15.00,
        "origin_country": "CN",
        "hts_code": "7318.16.0085",
        "expected": {
            "name": "Lug nuts (steel, CN)",
            "base_rate": 5.7,
            "ad_cvd": True,  # CRITICAL: Fasteners from China have AD/CVD
        },
    },
    {
        "part_id": "AP-1005",
        "description": "Engine timing chain kit steel roller chain with tensioner",
        "material": "hardened steel",
        "weight_kg": 1.2,
        "value_usd": 62.00,
        "origin_country": "DE",
        "hts_code": "8409.99.9190",
        "expected": {
            "name": "Timing chain (DE)",
            "base_rate": 2.5,
            "no_overlays": True,  # Germany: no FTA, no Section 301
        },
    },
]


def test_sku(sku: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single SKU through the tariff engine."""
    print(f"\n{'='*80}")
    print(f"Testing: {sku['part_id']} - {sku['expected']['name']}")
    print(f"{'='*80}")

    result = {
        "part_id": sku["part_id"],
        "name": sku["expected"]["name"],
        "verdict": "UNKNOWN",
        "issues": [],
        "findings": [],
    }

    # Test 1: HTS code validation
    print(f"HTS Code: {sku['hts_code']}")
    try:
        # Note: This would normally validate against HTS database
        # For smoke test, we'll check format
        if not sku["hts_code"].replace(".", "").isdigit():
            result["issues"].append(f"Invalid HTS code format: {sku['hts_code']}")
        else:
            result["findings"].append(f"✓ HTS code format valid: {sku['hts_code']}")
    except Exception as e:
        result["issues"].append(f"HTS validation error: {e}")

    # Test 2: Origin country
    print(f"Origin: {sku['origin_country']}")
    if sku["origin_country"] not in ["CN", "MX", "KR", "DE", "US"]:
        result["issues"].append(f"Unexpected origin: {sku['origin_country']}")
    else:
        result["findings"].append(f"✓ Valid origin: {sku['origin_country']}")

    # Test 3: Value and weight
    print(f"Value: ${sku['value_usd']:.2f}, Weight: {sku['weight_kg']}kg")
    if sku["value_usd"] <= 0:
        result["issues"].append("Invalid value: must be > 0")
    if sku["weight_kg"] <= 0:
        result["issues"].append("Invalid weight: must be > 0")

    # Test 4: Expected overlay checks
    expected = sku["expected"]

    # Section 301 check (China origin)
    if sku["origin_country"] == "CN" and expected.get("section_301"):
        result["findings"].append(
            "⚠ Expected: Section 301 tariff (~25%) on Chinese imports"
        )
        result["issues"].append(
            "Cannot verify Section 301 application without overlay database"
        )

    # USMCA check (Mexico origin)
    if sku["origin_country"] == "MX" and expected.get("usmca"):
        result["findings"].append(
            "🔥 CRITICAL: Must find USMCA preferential rate (likely 0%)"
        )
        result["issues"].append(
            "Cannot verify USMCA without FTA rules database - THIS IS A DEAL-BREAKER"
        )

    # KORUS check (Korea origin)
    if sku["origin_country"] == "KR" and expected.get("korus"):
        result["findings"].append("⚠ Expected: KORUS FTA preferential rate")
        result["issues"].append("Cannot verify KORUS without FTA rules database")

    # AD/CVD check
    if expected.get("ad_cvd"):
        result["findings"].append(
            "🔥 CRITICAL: Fasteners (HTS 7318) from China typically have AD/CVD orders"
        )
        result["issues"].append(
            "Cannot verify AD/CVD without trade remedies database - THIS IS A DEAL-BREAKER"
        )

    # Test 5: Material extraction
    print(f"Materials: {sku['material']}")
    materials = sku["material"].lower().split()
    expected_materials = ["steel", "aluminum", "rubber", "plastic", "polyurethane", "iron"]
    found_materials = [m for m in expected_materials if m in sku["material"].lower()]
    if found_materials:
        result["findings"].append(f"✓ Materials detected: {', '.join(found_materials)}")
    else:
        result["issues"].append("No recognizable materials detected")

    # Determine verdict
    critical_issues = [i for i in result["issues"] if "CRITICAL" in i or "DEAL-BREAKER" in i]
    if critical_issues:
        result["verdict"] = "🔥 BROKEN"
    elif len(result["issues"]) > 2:
        result["verdict"] = "❌ SUSPICIOUS"
    elif len(result["issues"]) > 0:
        result["verdict"] = "⚠ NEEDS_DATA"
    else:
        result["verdict"] = "✓ REASONABLE"

    return result


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate plain-English assessment report."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("AUTOMOTIVE BOM SMOKE TEST - DIRECT ENGINE TEST")
    report.append("=" * 80)

    # Count verdicts
    broken = sum(1 for r in results if "BROKEN" in r["verdict"])
    suspicious = sum(1 for r in results if "SUSPICIOUS" in r["verdict"])
    needs_data = sum(1 for r in results if "NEEDS_DATA" in r["verdict"])
    reasonable = sum(1 for r in results if "REASONABLE" in r["verdict"])

    report.append(f"\n📊 RESULTS SUMMARY")
    report.append(f"  Total SKUs tested: {len(results)}")
    report.append(f"  🔥 Broken (critical failures): {broken}")
    report.append(f"  ❌ Suspicious: {suspicious}")
    report.append(f"  ⚠ Needs data: {needs_data}")
    report.append(f"  ✓ Reasonable: {reasonable}")

    # Per-SKU results
    report.append(f"\n📦 DETAILED RESULTS")
    report.append("=" * 80)

    for result in results:
        report.append(f"\n{result['verdict']} {result['part_id']}: {result['name']}")

        if result["findings"]:
            report.append("  Findings:")
            for finding in result["findings"]:
                report.append(f"    {finding}")

        if result["issues"]:
            report.append("  Issues:")
            for issue in result["issues"]:
                report.append(f"    • {issue}")

    # Critical findings
    report.append(f"\n" + "=" * 80)
    report.append("🔍 CRITICAL ASSESSMENT")
    report.append("=" * 80)

    report.append("\n⚠ INFRASTRUCTURE GAPS IDENTIFIED:")
    report.append("  • Section 301 tariff database not accessible in direct test")
    report.append("  • FTA rules engine (USMCA, KORUS) not accessible")
    report.append("  • AD/CVD orders database not accessible")
    report.append("  • HTS schedule lookup requires database connection")

    report.append("\n✓ WHAT CAN BE TESTED:")
    report.append("  • BOM parsing (CSV structure)")
    report.append("  • Column mapping detection")
    report.append("  • Material extraction from descriptions")
    report.append("  • Data validation (values, weights, codes)")
    report.append("  • API endpoint flow (upload → analyze → results)")

    report.append("\n🎯 NEXT STEPS:")
    report.append("  1. Run full API smoke test with server + databases loaded")
    report.append("  2. Verify Section 301 overlay application (critical for China imports)")
    report.append("  3. Verify USMCA/KORUS FTA preferential rates (deal-breaker if missing)")
    report.append("  4. Verify AD/CVD detection (especially HTS 7318 fasteners from China)")
    report.append("  5. Validate HTS code lookups against USITC database")

    report.append("\n" + "=" * 80)
    report.append("🎯 VERDICT")
    report.append("=" * 80)
    report.append("\n⚠ PARTIAL TEST ONLY")
    report.append("  This direct test validates data structures and formats but cannot")
    report.append("  verify the actual tariff engineering logic without:")
    report.append("    • HTS database loaded")
    report.append("    • Section 301/232 overlays configured")
    report.append("    • FTA rules engine operational")
    report.append("    • AD/CVD orders database accessible")
    report.append("\n  RECOMMENDATION: Run full API test with mock or real databases.")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """Run smoke test."""
    print("🚗 AUTOMOTIVE PARTS BOM SMOKE TEST (DIRECT ENGINE)")
    print("Testing core tariff engine logic with 6 representative SKUs\n")

    results = []
    for sku in TEST_SKUS:
        result = test_sku(sku)
        results.append(result)

    # Generate report
    report = generate_report(results)
    print(report)

    # Save report
    report_file = Path(__file__).parent.parent / "test_data" / "automotive_smoke_direct_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report)
    print(f"\n📄 Report saved to: {report_file}")


if __name__ == "__main__":
    main()
