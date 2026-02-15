from __future__ import annotations

import json
from pathlib import Path

from potatobacon.tariff.adcvd_registry import get_adcvd_registry
from potatobacon.tariff.auto_classifier import classify_product
from potatobacon.tariff.duty_calculator import compute_total_duty
from potatobacon.tariff.rate_store import get_rate_store
from potatobacon.tariff.strategy_optimizer import optimize_sku


DATA_PATH = Path(__file__).resolve().parents[1] / "test_data" / "stress_test_50_skus.json"


def _digits(code: str | None) -> str:
    return "".join(ch for ch in str(code or "") if ch.isdigit())


def _chapter(code: str | None) -> int | None:
    digits = _digits(code)
    if len(digits) < 2:
        return None
    return int(digits[:2])


def _expected_chapters(row: dict) -> set[int]:
    industry = row["industry"]
    if industry == "consumer_electronics":
        return {84, 85, 90}
    if industry == "textiles_apparel":
        return set(range(41, 64))
    if industry == "industrial_machinery":
        return {84, 85}
    if industry == "plastics_rubber":
        return {39, 40}
    if industry == "steel_metals":
        return {72, 73, 74, 75, 76, 81}
    if industry == "furniture":
        return {94}
    if industry == "chemicals":
        return set(range(28, 39))
    if industry == "food_agriculture":
        return set(range(1, 25))
    if industry == "solar_energy":
        return {84, 85, 94}
    if industry == "auto_parts":
        return {73, 85, 87}
    return set(range(1, 100))


def test_stress_50_skus_end_to_end():
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8-sig"))
    assert len(rows) == 50

    registry = get_adcvd_registry()
    rate_store = get_rate_store()

    duty_success = 0
    auto_total = 0
    auto_chapter_ok = 0
    cn_total = 0
    cn_301_hits = 0
    cn_301_expected = 0
    cn_301_misses = 0
    strategy_counts = {"origin_shift": 0, "reclassification": 0, "product_modification": 0, "first_sale": 0, "drawback": 0}
    dangerous: list[str] = []
    top_savings: list[tuple[float, str, str]] = []
    adcvd_known_hits = 0

    required_adcvd = {
        "FA-8001": "A-570-863",
        "FA-8002": "A-552-802",
        "SE-9001": "A-570-979",
        "SM-5002": "A-570-967",
    }

    for row in rows:
        materials = [token.strip() for token in str(row["material"]).split(",")]
        hts_code = row.get("hts_code")
        declared_or_probe_hts = hts_code

        if hts_code is None:
            auto_total += 1
            candidates = classify_product(
                description=row["description"],
                materials=materials,
                weight_kg=float(row["weight_kg"]),
                value_usd=float(row["value_usd"]),
            )
            assert candidates, f"No classification candidates for {row['part_id']}"
            top = candidates[0]
            hts_code = top.hts_code
            if top.chapter in _expected_chapters(row):
                auto_chapter_ok += 1
        elif not rate_store.lookup(hts_code).found:
            # Resilient fallback: keep declared code for AD/CVD probing, but classify for duty math.
            candidates = classify_product(
                description=row["description"],
                materials=materials,
                weight_kg=float(row["weight_kg"]),
                value_usd=float(row["value_usd"]),
            )
            if candidates:
                hts_code = candidates[0].hts_code

        try:
            duty = compute_total_duty(
                hts_code=hts_code,
                origin_country=row["origin_country"],
                import_country="US",
                declared_value=float(row["value_usd"]),
                weight_kg=float(row["weight_kg"]),
            )
            duty_success += 1
        except Exception as exc:  # pragma: no cover - surfaced by aggregate assertions
            dangerous.append(f"{row['part_id']} duty error: {exc}")
            continue

        if row["origin_country"] == "CN":
            cn_total += 1
            chapter = _chapter(hts_code)
            expects_301 = chapter is None or not (1 <= chapter <= 24)
            if expects_301:
                cn_301_expected += 1
            if duty.section_301_rate > 0:
                cn_301_hits += 1
            elif expects_301:
                cn_301_misses += 1

        result = optimize_sku(
            hts_code=hts_code,
            description=row["description"],
            materials=materials,
            value_usd=float(row["value_usd"]),
            weight_kg=float(row["weight_kg"]),
            origin_country=row["origin_country"],
            annual_volume=int(row["annual_volume"]),
            sku_id=row["part_id"],
        )
        for strategy in result.strategies:
            if strategy.type in strategy_counts:
                strategy_counts[strategy.type] += 1
            top_savings.append((strategy.estimated_annual_savings, row["part_id"], strategy.type))

        known_case = required_adcvd.get(row["part_id"])
        if known_case:
            lookup = registry.lookup(declared_or_probe_hts or hts_code, row["origin_country"], row["description"])
            if any(match.order.case_number == known_case for match in lookup.order_matches):
                adcvd_known_hits += 1
            else:
                dangerous.append(f"{row['part_id']} missing expected AD/CVD case {known_case}")

        if row["origin_country"] != "CN" and duty.section_301_rate > 0:
            dangerous.append(f"{row['part_id']} non-CN SKU received Section 301")

    top_savings.sort(key=lambda item: -item[0])
    top_10 = top_savings[:10]

    # Success criteria
    assert duty_success >= 45, f"Duty calculation success below threshold: {duty_success}/50"
    assert auto_chapter_ok >= 18, f"Auto-classification chapter correctness below threshold: {auto_chapter_ok}/25"
    assert cn_301_misses <= 1, f"Section 301 missing on too many expected CN SKUs: misses={cn_301_misses}"
    assert adcvd_known_hits >= 3, f"Known AD/CVD coverage below threshold: {adcvd_known_hits}/4"
    strategy_types_with_hits = sum(1 for value in strategy_counts.values() if value > 0)
    assert strategy_types_with_hits >= 2, f"Strategy diversity too low: {strategy_counts}"
    assert not dangerous, f"Dangerous advice/errors detected: {dangerous}"

    print(
        "Stress Test Summary: "
        f"duty {duty_success}/50, "
        f"auto-class chapter-correct {auto_chapter_ok}/{auto_total}, "
        f"CN 301 hits {cn_301_hits}/{cn_total}, "
        f"known AD/CVD {adcvd_known_hits}/4, "
        f"strategy_counts={strategy_counts}, "
        f"top10={top_10}"
    )
