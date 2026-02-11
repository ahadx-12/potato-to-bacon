#!/usr/bin/env python3
"""
Real-World Automotive BOM Smoke Test

Runs the complete tariff engineering pipeline on a realistic 12-SKU automotive
parts BOM and produces a critical plain-English assessment of the results.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

# Configuration
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
API_KEY = os.environ.get("PTB_API_KEY", "dev-key-local")
BOM_FILE = Path(__file__).parent.parent / "test_data" / "automotive_bom_smoke_test.csv"

# Expected reality checks (approximate)
REALITY_CHECK = {
    "AP-1001": {
        "name": "Brake rotor (stainless steel, CN)",
        "expected_base": 2.5,
        "expected_301": 25.0,
        "expected_total": 27.5,
        "notes": "Should show Section 301 List 3. Possible AD/CVD on steel from China.",
    },
    "AP-1002": {
        "name": "Aluminum wheel (CN)",
        "expected_base": 2.5,
        "expected_301": 25.0,
        "expected_232": 10.0,  # Section 232 aluminum
        "expected_total": 37.5,
        "notes": "Heavily tariffed: base + 301 + possibly 232 aluminum. Big savings opportunity.",
    },
    "AP-1003": {
        "name": "Brake hose (rubber/steel, CN)",
        "expected_base": 2.5,
        "expected_301": 25.0,
        "notes": "Should show Section 301.",
    },
    "AP-1004": {
        "name": "LED headlamp (CN)",
        "expected_base": 2.5,
        "expected_301": 25.0,
        "notes": "Electronics from China = Section 301 applies.",
    },
    "AP-1005": {
        "name": "Timing chain (steel, DE)",
        "expected_base": 2.5,
        "expected_total": 2.5,
        "notes": "Germany origin, no FTA, no 301. Clean MFN rate. Limited optimization.",
    },
    "AP-1006": {
        "name": "Control arm bushing (polyurethane, KR)",
        "expected_base": 2.5,
        "expected_korus": 0.0,
        "notes": "KORUS FTA should apply. Auto parts generally eligible.",
    },
    "AP-1007": {
        "name": "Exhaust manifold (cast iron, CN)",
        "expected_base": 2.5,
        "expected_301": 25.0,
        "notes": "Cast iron from China may trigger AD/CVD.",
    },
    "AP-1008": {
        "name": "Wiper motor (MX)",
        "expected_base": 2.7,
        "expected_usmca": 0.0,
        "notes": "CRITICAL: USMCA preference should apply. If not found, system is broken.",
    },
    "AP-1009": {
        "name": "Cabin air filter (CN)",
        "expected_base": 2.5,
        "expected_301": 7.5,  # List 4A (7.5%)
        "notes": "Section 301 List 4A (7.5% not 25%).",
    },
    "AP-1010": {
        "name": "Oil cooler (aluminum, CN)",
        "expected_base": 2.5,
        "expected_301": 25.0,
        "notes": "Section 301 + possible 232 aluminum.",
    },
    "AP-1011": {
        "name": "Lug nuts (steel, CN)",
        "expected_base": 5.7,
        "expected_adcvd": True,
        "notes": "CRITICAL: HTS 7318 fasteners from China have AD/CVD orders. System MUST find this.",
    },
    "AP-1012": {
        "name": "Ignition coil (KR)",
        "expected_base": 2.5,
        "expected_korus": 0.0,
        "notes": "KORUS FTA should apply.",
    },
}


def upload_bom() -> Dict[str, Any]:
    """Step 1: Upload BOM file."""
    print("=" * 80)
    print("STEP 1: Uploading BOM")
    print("=" * 80)

    with open(BOM_FILE, "rb") as f:
        files = {"file": ("automotive_bom.csv", f, "text/csv")}
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(f"{API_BASE}/v1/bom/upload", files=files, headers=headers, timeout=60)

    if resp.status_code != 200:
        print(f"❌ Upload failed: {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    data = resp.json()
    print(f"✓ Upload ID: {data['upload_id']}")
    print(f"✓ Status: {data['status']}")
    print(f"✓ Total rows: {data['total_rows']}")
    print(f"✓ Valid rows: {data['valid_rows']}")
    print(f"✓ Skipped rows: {data['skipped_rows']}")

    if data["skipped_rows"] > 0:
        print(f"⚠ Skipped rows:")
        for skip in data["skipped"]:
            print(f"  Row {skip['row_number']}: {skip['reason']}")

    print(f"\n✓ Column mapping detected:")
    for key, val in data["column_mapping"].items():
        print(f"  {key}: {val}")

    return data


def start_analysis(upload_id: str) -> Dict[str, Any]:
    """Step 2: Start batch analysis."""
    print("\n" + "=" * 80)
    print("STEP 2: Starting Batch Analysis")
    print("=" * 80)

    headers = {"X-API-Key": API_KEY}
    payload = {
        "law_context": "us_import_2025",
        "max_mutations": 3,
        "import_country": "US",
    }
    resp = requests.post(
        f"{API_BASE}/v1/bom/{upload_id}/analyze", json=payload, headers=headers, timeout=60
    )

    if resp.status_code != 200:
        print(f"❌ Analysis start failed: {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    data = resp.json()
    print(f"✓ Job ID: {data['job_id']}")
    print(f"✓ Status: {data['status']}")
    print(f"✓ Total items: {data['total_items']}")
    print(f"✓ Message: {data['message']}")

    return data


def poll_job(job_id: str, timeout: int = 600) -> Dict[str, Any]:
    """Step 3: Poll until job completes."""
    print("\n" + "=" * 80)
    print("STEP 3: Polling Job Status")
    print("=" * 80)

    headers = {"X-API-Key": API_KEY}
    start_time = time.time()

    while time.time() - start_time < timeout:
        resp = requests.get(f"{API_BASE}/v1/bom/{job_id}/status", headers=headers, timeout=30)

        if resp.status_code != 200:
            print(f"❌ Status check failed: {resp.status_code}")
            sys.exit(1)

        data = resp.json()
        status = data["status"]
        completed = data.get("completed", 0)
        total = data.get("total", 0)
        failed = data.get("failed", 0)

        print(f"\r[{status}] {completed}/{total} completed, {failed} failed", end="", flush=True)

        if status == "completed":
            print("\n✓ Job completed!")
            return data
        elif status == "failed":
            print("\n❌ Job failed!")
            print(f"Errors: {data.get('errors', [])}")
            sys.exit(1)

        time.sleep(2)

    print("\n❌ Timeout waiting for job completion")
    sys.exit(1)


def get_results(job_id: str) -> Dict[str, Any]:
    """Step 4: Retrieve full results."""
    print("\n" + "=" * 80)
    print("STEP 4: Retrieving Results")
    print("=" * 80)

    headers = {"X-API-Key": API_KEY}
    resp = requests.get(f"{API_BASE}/v1/bom/{job_id}/results", headers=headers, timeout=60)

    if resp.status_code != 200:
        print(f"❌ Results fetch failed: {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    data = resp.json()
    print(f"✓ Retrieved {len(data['results'])} dossiers")
    return data


def analyze_sku(sku: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single SKU's results against reality."""
    part_id = sku.get("sku_id") or sku.get("part_id")
    expected = REALITY_CHECK.get(part_id, {})

    analysis = {
        "part_id": part_id,
        "name": expected.get("name", "Unknown"),
        "baseline_found": None,
        "optimized_found": None,
        "savings_pct": None,
        "overlays_found": [],
        "verdict": "UNKNOWN",
        "issues": [],
        "notes": [],
    }

    # Extract baseline
    baseline = sku.get("baseline", {})
    if baseline:
        analysis["baseline_found"] = baseline.get("total_duty_rate")
        analysis["overlays_found"] = baseline.get("overlays_applied", [])

    # Extract optimized
    optimized = sku.get("optimized", {})
    if optimized:
        analysis["optimized_found"] = optimized.get("total_duty_rate")

    # Calculate savings
    if sku.get("savings"):
        analysis["savings_pct"] = sku["savings"].get("percentage")

    # Reality check
    if "expected_total" in expected:
        if analysis["baseline_found"]:
            diff = abs(analysis["baseline_found"] - expected["expected_total"])
            if diff > 5.0:  # >5% difference
                analysis["issues"].append(
                    f"Baseline rate {analysis['baseline_found']:.1f}% differs significantly from expected {expected['expected_total']:.1f}%"
                )
                analysis["verdict"] = "SUSPICIOUS"
            else:
                analysis["verdict"] = "REASONABLE"

    # Check for expected overlays
    if "expected_301" in expected:
        found_301 = any("301" in str(o).lower() for o in analysis["overlays_found"])
        if not found_301:
            analysis["issues"].append("Missing Section 301 tariff (expected)")
            analysis["verdict"] = "WRONG"

    if "expected_usmca" in expected:
        found_usmca = any("usmca" in str(o).lower() for o in analysis["overlays_found"])
        if not found_usmca:
            analysis["issues"].append("CRITICAL: Missing USMCA preference (deal-breaker)")
            analysis["verdict"] = "BROKEN"

    if "expected_korus" in expected:
        found_korus = any("korus" in str(o).lower() or "korea" in str(o).lower() for o in analysis["overlays_found"])
        if not found_korus:
            analysis["issues"].append("Missing KORUS FTA preference")
            analysis["verdict"] = "WRONG"

    if expected.get("expected_adcvd"):
        found_adcvd = any("ad" in str(o).lower() or "cvd" in str(o).lower() or "antidumping" in str(o).lower() for o in analysis["overlays_found"])
        if not found_adcvd:
            analysis["issues"].append("CRITICAL: Missing AD/CVD order (lug nuts from China)")
            analysis["verdict"] = "BROKEN"

    analysis["notes"].append(expected.get("notes", ""))

    return analysis


def generate_report(results: Dict[str, Any], analyses: List[Dict[str, Any]]) -> str:
    """Generate comprehensive plain-English report."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("REAL-WORLD AUTOMOTIVE BOM SMOKE TEST - CRITICAL ASSESSMENT")
    report.append("=" * 80)

    # Overall stats
    total_skus = len(results["results"])
    completed = sum(1 for a in analyses if a["verdict"] not in ["UNKNOWN", "BROKEN"])
    broken = sum(1 for a in analyses if a["verdict"] == "BROKEN")
    suspicious = sum(1 for a in analyses if a["verdict"] == "SUSPICIOUS")
    reasonable = sum(1 for a in analyses if a["verdict"] == "REASONABLE")

    report.append(f"\n📊 OVERALL RESULTS")
    report.append(f"  Total SKUs: {total_skus}")
    report.append(f"  ✓ Reasonable: {reasonable}")
    report.append(f"  ⚠ Suspicious: {suspicious}")
    report.append(f"  ❌ Broken: {broken}")

    # Portfolio summary
    if results.get("summary"):
        summary = results["summary"]
        report.append(f"\n💰 PORTFOLIO SUMMARY")
        report.append(f"  Baseline annual duty: ${summary.get('baseline_total_annual_duty', 0):,.2f}")
        report.append(f"  Optimized annual duty: ${summary.get('optimized_total_annual_duty', 0):,.2f}")
        report.append(f"  Annual savings: ${summary.get('total_annual_savings', 0):,.2f}")
        report.append(f"  Savings percentage: {summary.get('savings_percentage', 0):.1f}%")

    # Per-SKU analysis
    report.append(f"\n📦 PER-SKU ANALYSIS")
    report.append("=" * 80)

    for analysis in analyses:
        verdict_emoji = {
            "REASONABLE": "✓",
            "SUSPICIOUS": "⚠",
            "WRONG": "❌",
            "BROKEN": "🔥",
            "UNKNOWN": "?",
        }.get(analysis["verdict"], "?")

        report.append(f"\n{verdict_emoji} {analysis['part_id']}: {analysis['name']}")
        report.append(f"  Baseline: {analysis['baseline_found']:.1f}% duty" if analysis['baseline_found'] else "  Baseline: Not found")
        report.append(f"  Optimized: {analysis['optimized_found']:.1f}% duty" if analysis['optimized_found'] else "  Optimized: None")
        report.append(f"  Savings: {analysis['savings_pct']:.1f}%" if analysis['savings_pct'] else "  Savings: None")
        report.append(f"  Overlays: {', '.join(analysis['overlays_found']) if analysis['overlays_found'] else 'None'}")

        if analysis["issues"]:
            report.append(f"  🔍 ISSUES:")
            for issue in analysis["issues"]:
                report.append(f"    • {issue}")

        if analysis["notes"] and analysis["notes"][0]:
            report.append(f"  📝 {analysis['notes'][0]}")

    # Critical findings
    report.append(f"\n" + "=" * 80)
    report.append("🔍 CRITICAL FINDINGS")
    report.append("=" * 80)

    # Check for deal-breakers
    deal_breakers = [a for a in analyses if a["verdict"] == "BROKEN"]
    if deal_breakers:
        report.append(f"\n🔥 DEAL-BREAKERS (system is broken):")
        for a in deal_breakers:
            report.append(f"  • {a['part_id']}: {', '.join(a['issues'])}")

    # Check for suspicious results
    suspicious_items = [a for a in analyses if a["verdict"] == "SUSPICIOUS"]
    if suspicious_items:
        report.append(f"\n⚠ SUSPICIOUS RESULTS (needs review):")
        for a in suspicious_items:
            report.append(f"  • {a['part_id']}: {', '.join(a['issues'])}")

    # What worked
    good_items = [a for a in analyses if a["verdict"] == "REASONABLE"]
    if good_items:
        report.append(f"\n✓ WHAT WORKED WELL:")
        for a in good_items:
            report.append(f"  • {a['part_id']}: Realistic duty calculation")

    # Final verdict
    report.append(f"\n" + "=" * 80)
    report.append("🎯 FINAL VERDICT")
    report.append("=" * 80)

    if broken > 0:
        report.append(f"\n❌ SYSTEM NOT READY FOR PRODUCTION")
        report.append(f"  {broken} critical failures found. An actual importer would NOT trust this output.")
        report.append(f"  Major issues must be fixed before showing to customers.")
    elif suspicious > 2:
        report.append(f"\n⚠ SYSTEM NEEDS WORK")
        report.append(f"  {suspicious} suspicious results. System is functional but not yet reliable.")
        report.append(f"  Would need customs broker review before trusting.")
    else:
        report.append(f"\n✓ SYSTEM SHOWS PROMISE")
        report.append(f"  {reasonable}/{total_skus} SKUs produced reasonable results.")
        report.append(f"  An importer would likely trust this as a starting point for customs broker review.")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """Run complete smoke test."""
    print("🚗 AUTOMOTIVE PARTS BOM SMOKE TEST")
    print("Testing complete tariff engineering pipeline with 12 realistic SKUs\n")

    # Step 1: Upload BOM
    upload_data = upload_bom()
    upload_id = upload_data["upload_id"]

    # Validate upload
    if upload_data["valid_rows"] != 12:
        print(f"❌ Expected 12 valid rows, got {upload_data['valid_rows']}")
        sys.exit(1)

    # Step 2: Start analysis
    analysis_data = start_analysis(upload_id)
    job_id = analysis_data["job_id"]

    # Step 3: Poll for completion
    status_data = poll_job(job_id, timeout=600)

    # Step 4: Get results
    results = get_results(job_id)

    # Step 5: Analyze each SKU
    print("\n" + "=" * 80)
    print("STEP 5: Analyzing Results")
    print("=" * 80)

    analyses = []
    for sku_result in results["results"]:
        analysis = analyze_sku(sku_result)
        analyses.append(analysis)

    # Step 6: Generate report
    report = generate_report(results, analyses)
    print(report)

    # Save report to file
    report_file = Path(__file__).parent.parent / "test_data" / "automotive_smoke_test_report.txt"
    report_file.write_text(report)
    print(f"\n📄 Full report saved to: {report_file}")


if __name__ == "__main__":
    main()
