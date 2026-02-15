from __future__ import annotations

from potatobacon.tariff.strategy_optimizer import optimize_portfolio, optimize_sku


def test_optimize_sku_emits_multiple_strategy_types():
    result = optimize_sku(
        hts_code="8507.60.00",
        description="Lithium-ion battery module for portable electronics",
        materials=["battery", "electronics"],
        value_usd=78.0,
        weight_kg=0.45,
        origin_country="CN",
        annual_volume=5000,
        sku_id="CE-TEST-1",
    )
    assert result.strategies
    strategy_types = {item.type for item in result.strategies}
    assert "origin_shift" in strategy_types
    assert "first_sale" in strategy_types


def test_optimize_portfolio_aggregates():
    portfolio = optimize_portfolio(
        [
            {
                "part_id": "SKU-A",
                "description": "Lithium-ion battery module",
                "material": "battery, electronics",
                "value_usd": 78.0,
                "weight_kg": 0.45,
                "origin_country": "CN",
                "hts_code": "8507.60.00",
                "annual_volume": 18000,
            },
            {
                "part_id": "SKU-B",
                "description": "Wiper motor assembly",
                "material": "electronics, steel",
                "value_usd": 36.0,
                "weight_kg": 1.7,
                "origin_country": "DE",
                "hts_code": "8501.31.40",
                "annual_volume": 9000,
            },
        ]
    )
    assert portfolio.total_current_duty >= 0
    assert portfolio.total_optimized_duty >= 0
    assert portfolio.top_10_strategies
