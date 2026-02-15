#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from potatobacon.tariff.models import TariffSuggestRequestModel
from potatobacon.tariff.suggest import suggest_tariff_optimizations


def _default_benchmark() -> list[dict[str, Any]]:
    seed_rows = [
        ("auto_parts", "AP-1001", "Chassis bolt fastener for EV frame", "steel", "7318.15", "CN"),
        ("auto_parts", "AP-1002", "Suspension control arm assembly", "steel", "8708.80", "MX"),
        ("auto_parts", "AP-1003", "Brake rotor cast iron disc", "iron", "8708.30", "CN"),
        ("auto_parts", "AP-1004", "Automotive wiring harness with connectors", "copper pvc", "8544.42", "VN"),
        ("auto_parts", "AP-1005", "Airbag module inflator assembly", "metal textile", "8708.95", "JP"),
        ("consumer_electronics", "CE-2001", "Laptop computer portable", "electronics", "8471.30", "CN"),
        ("consumer_electronics", "CE-2002", "Phone case molded plastic", "plastic", "3926.90", "VN"),
        ("consumer_electronics", "CE-2003", "USB cable assembly", "copper plastic", "8544.42", "KR"),
        ("consumer_electronics", "CE-2004", "Bluetooth speaker portable", "electronics", "8518.22", "CN"),
        ("consumer_electronics", "CE-2005", "Power adapter AC/DC", "electronics", "8504.40", "CN"),
        ("textiles_apparel", "TX-3001", "Men cotton shirt woven", "cotton textile", "6205.20", "VN"),
        ("textiles_apparel", "TX-3002", "Polyester jacket outerwear", "polyester textile", "6201.93", "BD"),
        ("textiles_apparel", "TX-3003", "Leather belt accessory", "leather", "4203.30", "CN"),
        ("textiles_apparel", "TX-3004", "Wool yarn combed", "wool", "5107.10", "IN"),
        ("textiles_apparel", "TX-3005", "Synthetic woven fabric roll", "polyester textile", "5407.52", "VN"),
        ("industrial_machinery", "IM-4001", "CNC spindle motor unit", "machinery", "8466.93", "CN"),
        ("industrial_machinery", "IM-4002", "Hydraulic pump industrial", "steel", "8413.50", "DE"),
        ("industrial_machinery", "IM-4003", "Conveyor bearing assembly", "steel", "8482.10", "JP"),
        ("industrial_machinery", "IM-4004", "Industrial valve body", "steel", "8481.80", "TW"),
        ("industrial_machinery", "IM-4005", "Servo motor precision", "electronics", "8501.52", "JP"),
        ("plastics_rubber", "PR-5001", "PVC pipe fitting elbow", "plastic", "3917.40", "CN"),
        ("plastics_rubber", "PR-5002", "Silicone gasket ring", "rubber", "4016.93", "MY"),
        ("plastics_rubber", "PR-5003", "Polyethylene packaging film", "plastic", "3920.10", "TH"),
        ("plastics_rubber", "PR-5004", "Rubber gloves disposable", "rubber", "4015.19", "MY"),
        ("plastics_rubber", "PR-5005", "Acrylic sheet panel", "plastic", "3920.51", "CN"),
        ("steel_metals", "SM-6001", "Stainless steel sheet", "stainless steel", "7219.33", "KR"),
        ("steel_metals", "SM-6002", "Aluminum extrusion profile", "aluminum", "7604.29", "IN"),
        ("steel_metals", "SM-6003", "Copper wire insulated", "copper", "8544.49", "DE"),
        ("steel_metals", "SM-6004", "Brass plumbing fitting", "brass", "7412.20", "CN"),
        ("steel_metals", "SM-6005", "Titanium machine fastener", "titanium", "8108.90", "DE"),
        ("furniture", "FU-7001", "Wooden office desk", "wood", "9403.30", "CN"),
        ("furniture", "FU-7002", "Upholstered sofa seat", "textile wood", "9401.61", "VN"),
        ("furniture", "FU-7003", "Metal shelving unit", "steel", "9403.20", "MX"),
        ("furniture", "FU-7004", "Mattress foam spring", "foam textile", "9404.21", "CN"),
        ("furniture", "FU-7005", "Office chair swivel", "steel textile", "9401.30", "VN"),
        ("chemicals", "CH-8001", "Epoxy resin industrial", "chemical", "3907.30", "CN"),
        ("chemicals", "CH-8002", "Industrial solvent blend", "chemical", "3814.00", "DE"),
        ("chemicals", "CH-8003", "Paint pigment concentrate", "chemical", "3204.17", "JP"),
        ("chemicals", "CH-8004", "Adhesive formulation", "chemical", "3506.91", "CN"),
        ("chemicals", "CH-8005", "Lubricating preparation", "chemical", "3403.19", "JP"),
        ("food_agriculture", "FA-9001", "Natural honey packed", "food", "0409.00", "IN"),
        ("food_agriculture", "FA-9002", "Frozen shrimp peeled", "food", "0306.17", "VN"),
        ("food_agriculture", "FA-9003", "Olive oil extra virgin", "food", "1509.20", "IT"),
        ("food_agriculture", "FA-9004", "Coffee beans green", "food", "0901.11", "CO"),
        ("food_agriculture", "FA-9005", "Canned tomatoes whole", "food", "2002.10", "IT"),
        ("solar_energy", "SE-10001", "Solar photovoltaic panel", "electronics", "8541.43", "MY"),
        ("solar_energy", "SE-10002", "Solar inverter", "electronics", "8504.40", "CN"),
        ("solar_energy", "SE-10003", "Lithium battery module", "battery", "8507.60", "KR"),
        ("solar_energy", "SE-10004", "Wind turbine bearing", "steel", "8482.10", "VN"),
        ("solar_energy", "SE-10005", "LED lighting fixture", "electronics", "9405.40", "CN"),
    ]
    rows: list[dict[str, Any]] = []
    for index, (industry, sku_id, description, material, hts_code, origin) in enumerate(seed_rows):
        include_declared = index % 2 == 0
        row = {
            "industry": industry,
            "sku_id": sku_id,
            "description": description,
            "material": material,
            "weight_kg": round(0.8 + (index % 7) * 0.45, 2),
            "declared_value_per_unit": round(8.0 + (index % 9) * 7.5, 2),
            "annual_volume": 5000 + (index % 10) * 1500,
            "origin_country": origin,
            "reference_hts_code": hts_code,
            "bom_json": {
                "items": [
                    {
                        "part_id": f"{sku_id}-01",
                        "description": description,
                        "material": material,
                        "quantity": 1,
                        "unit_cost": round(3.5 + (index % 6), 2),
                        "weight_kg": round(0.4 + (index % 5) * 0.3, 2),
                        "country_of_origin": origin,
                        "hts_code": hts_code if include_declared else None,
                    }
                ],
                "currency": "USD",
            },
        }
        rows.append(row)
    return rows


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return _default_benchmark()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark file must be a JSON array.")
    return payload


def _is_reasonable_classification(predicted: str | None, reference: str | None) -> bool:
    if not predicted or not reference:
        return False
    pred = "".join(ch for ch in predicted if ch.isdigit())
    ref = "".join(ch for ch in reference if ch.isdigit())
    if not pred or not ref:
        return False
    return pred[:4] == ref[:4] or pred[:2] == ref[:2]


def _top_savings(rows: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda item: -(item.get("annual_savings_value") or 0.0))
    return rows[:n]


def run_benchmark(path: Path) -> dict[str, Any]:
    rows = _load_json(path)
    results: list[dict[str, Any]] = []

    baseline_total = 0.0
    optimized_total = 0.0
    strategy_types: set[str] = set()
    dangerous: list[dict[str, Any]] = []
    auto_total = 0
    auto_reasonable = 0
    duty_reasonable = 0

    for row in rows:
        request = TariffSuggestRequestModel(
            sku_id=row["sku_id"],
            description=row["description"],
            declared_value_per_unit=row["declared_value_per_unit"],
            annual_volume=row["annual_volume"],
            origin_country=row.get("origin_country"),
            bom_json=row.get("bom_json"),
            top_k=10,
        )
        response = suggest_tariff_optimizations(request)
        top = response.suggestions[0] if response.suggestions else None
        baseline_rate = top.baseline_effective_duty_rate if top and top.baseline_effective_duty_rate is not None else (
            top.baseline_duty_rate if top else 0.0
        )
        optimized_rate = top.optimized_effective_duty_rate if top and top.optimized_effective_duty_rate is not None else (
            top.optimized_duty_rate if top else baseline_rate
        )
        value = row["declared_value_per_unit"] * row["annual_volume"]
        baseline_total += value * (baseline_rate or 0.0) / 100.0
        optimized_total += value * (optimized_rate or 0.0) / 100.0
        if baseline_rate is not None and optimized_rate is not None and optimized_rate >= 0:
            duty_reasonable += 1

        if response.auto_classification and response.auto_classification.hts_source == "auto_classified":
            auto_total += 1
            if _is_reasonable_classification(
                response.auto_classification.selected_hts_code,
                row.get("reference_hts_code"),
            ):
                auto_reasonable += 1

        for suggestion in response.suggestions:
            strategy_types.add(suggestion.strategy_type or suggestion.optimization_type or "unknown")
            if suggestion.risk_level == "high_risk" and "ad/cvd" in " ".join(suggestion.risk_reasons or []).lower():
                dangerous.append(
                    {
                        "sku_id": row["sku_id"],
                        "strategy": suggestion.strategy_type or suggestion.optimization_type,
                        "risk_reasons": suggestion.risk_reasons,
                    }
                )
            results.append(
                {
                    "sku_id": row["sku_id"],
                    "strategy_type": suggestion.strategy_type or suggestion.optimization_type,
                    "human_summary": suggestion.human_summary,
                    "annual_savings_value": suggestion.annual_savings_value or 0.0,
                    "risk_level": suggestion.risk_level,
                }
            )

    top10 = _top_savings(results, n=10)
    return {
        "total_skus": len(rows),
        "duty_reasonable_count": duty_reasonable,
        "auto_classified_total": auto_total,
        "auto_classified_reasonable": auto_reasonable,
        "baseline_total_duty": round(baseline_total, 2),
        "optimized_total_duty": round(optimized_total, 2),
        "total_savings": round(baseline_total - optimized_total, 2),
        "strategy_types_proposed": sorted(strategy_types),
        "dangerous_advice_candidates": dangerous,
        "top_10_savings_opportunities": top10,
        "verdict": (
            "universal_tariff_engineer"
            if {"origin_shift", "reclassification", "product_modification"} & strategy_types
            else "limited_scope"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-SKU stress benchmark across industries.")
    parser.add_argument(
        "--input",
        default="data/benchmarks/stress_50_skus.json",
        help="JSON benchmark file",
    )
    parser.add_argument(
        "--output",
        default="reports/stress_test_50_report.json",
        help="Output report path",
    )
    args = parser.parse_args()

    report = run_benchmark(Path(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
