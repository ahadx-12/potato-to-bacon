from __future__ import annotations

from potatobacon.tariff.hts_ingest.usitc_parser import parse_usitc_duty_rate
from potatobacon.tariff.rate_store import RateEntry
from potatobacon.tariff.reclassification_engine import (
    build_advisory_strategies,
    build_auto_classification_payload,
    build_reclassification_candidates,
)


def _entry(hts_code: str, description: str, general: str) -> RateEntry:
    return RateEntry(
        hts_code=hts_code,
        hts_digits="".join(ch for ch in hts_code if ch.isdigit()),
        description=description,
        general_rate=parse_usitc_duty_rate(general),
        special_rates={},
    )


def test_auto_classification_prefers_material_overlap(monkeypatch):
    fake_entries = [
        _entry("7326.90", "Other articles of iron or steel", "3.4%"),
        _entry("7326.19", "Other articles of stainless steel", "Free"),
        _entry("8413.50", "Hydraulic pumps", "2.5%"),
    ]
    monkeypatch.setattr(
        "potatobacon.tariff.reclassification_engine._rate_entries",
        lambda: fake_entries,
    )

    payload = build_auto_classification_payload(
        description="stainless shelf bracket",
        material="stainless steel",
        declared_hts=None,
    )
    assert payload["classification"] == "auto"
    assert payload["selected_hts_code"] in {"7326.19", "7326.90"}
    assert payload["alternatives"]


def test_declared_hts_mismatch_flag(monkeypatch):
    fake_entries = [
        _entry("8507.60", "Lithium-ion batteries", "3.4%"),
        _entry("0901.11", "Coffee, not roasted", "Free"),
    ]
    monkeypatch.setattr(
        "potatobacon.tariff.reclassification_engine._rate_entries",
        lambda: fake_entries,
    )

    payload = build_auto_classification_payload(
        description="lithium battery module",
        material="battery cells",
        declared_hts="0901.11",
    )
    assert payload["classification"] == "provided"
    assert payload["mismatch_flag"] is True
    assert payload["needs_manual_review"] is True


def test_reclassification_candidates_sorted_by_savings_and_risk(monkeypatch):
    fake_entries = [
        _entry("7326.90", "Other articles of iron or steel", "3.4%"),
        _entry("7326.19", "Other articles of stainless steel", "Free"),
        _entry("7616.99", "Other articles of aluminum", "2.5%"),
    ]
    monkeypatch.setattr(
        "potatobacon.tariff.reclassification_engine._rate_entries",
        lambda: fake_entries,
    )

    items = build_reclassification_candidates(
        current_hts="7326.90",
        baseline_rate=3.4,
        description="stainless steel bracket",
        material="stainless steel",
        annual_volume=1000,
        declared_value_per_unit=10.0,
    )
    assert items
    assert items[0]["to_hts"] == "7326.19"
    assert items[0]["risk_level"] in {"low_risk", "medium_risk"}


def test_advisory_strategies_include_first_sale_and_drawback():
    items = build_advisory_strategies(
        origin_country="CN",
        bom_items=[{"part_id": f"P-{idx}"} for idx in range(6)],
        baseline_rate=10.0,
        declared_value_per_unit=12.0,
        annual_volume=2000,
        material_breakdown=[
            {"material": "steel", "percent_by_weight": 51.0},
            {"material": "aluminum", "percent_by_weight": 49.0},
        ],
    )
    strategy_types = {item["strategy_type"] for item in items}
    assert "first_sale_valuation" in strategy_types
    assert "duty_drawback" in strategy_types
    assert "product_modification" in strategy_types
