import pytest

from potatobacon.cale.finance.numeric import extract_numeric_covenants


def _expect_single(text: str):
    results = extract_numeric_covenants(text)
    assert results, f"no covenant detected for: {text}"
    return results[0]


BASIC_CASES = [
    (
        "Total leverage ratio shall not exceed 5.25x.",
        {
            "metric": "LEVERAGE",
            "subtype": "TOTAL",
            "op": "<=",
            "value": 5.25,
            "unit": "x",
        },
    ),
    (
        "Net leverage will not exceed 4.0 to 1.",
        {
            "metric": "LEVERAGE",
            "subtype": "NET",
            "op": "<=",
            "value": 4.0,
            "unit": "x",
            "qualifiers": {"net": True},
        },
    ),
    (
        "Senior secured leverage shall not exceed five and one-half (5.5) to one.",
        {
            "metric": "LEVERAGE",
            "subtype": "SENIOR_SECURED",
            "op": "<=",
            "value": 5.5,
            "unit": "x",
            "qualifiers": {"secured": "SENIOR"},
        },
    ),
    (
        "First-lien net leverage shall not exceed 4.75 : 1.00.",
        {
            "metric": "LEVERAGE",
            "subtype": "FIRST_LIEN",
            "op": "<=",
            "value": 4.75,
            "unit": "x",
            "qualifiers": {"secured": "FIRST_LIEN", "net": True},
        },
    ),
    (
        "Consolidated net leverage may not exceed 4.25x.",
        {
            "metric": "LEVERAGE",
            "subtype": "NET",
            "op": "<=",
            "value": 4.25,
            "unit": "x",
            "qualifiers": {"consolidated": True, "net": True},
        },
    ),
    (
        "Total leverage must remain no greater than 6.0x.",
        {
            "metric": "LEVERAGE",
            "subtype": "TOTAL",
            "op": "<=",
            "value": 6.0,
            "unit": "x",
        },
    ),
    (
        "Total leverage is capped at 6.25x.",
        {
            "metric": "LEVERAGE",
            "subtype": "TOTAL",
            "op": "<=",
            "value": 6.25,
            "unit": "x",
        },
    ),
    (
        "Consolidated total leverage ratio shall not exceed 5.15 x.",
        {
            "metric": "LEVERAGE",
            "subtype": "TOTAL",
            "op": "<=",
            "value": 5.15,
            "unit": "x",
            "qualifiers": {"consolidated": True},
        },
    ),
    (
        "Total leverage shall not exceed 5 times.",
        {
            "metric": "LEVERAGE",
            "subtype": "TOTAL",
            "op": "<=",
            "value": 5.0,
            "unit": "x",
        },
    ),
    (
        "Total leverage ratio shall not exceed 5.00\u00a0x.",
        {
            "metric": "LEVERAGE",
            "subtype": "TOTAL",
            "op": "<=",
            "value": 5.0,
            "unit": "x",
        },
    ),
    (
        "Interest coverage shall not be less than 3.0x.",
        {
            "metric": "COVERAGE",
            "subtype": "INTEREST",
            "op": ">=",
            "value": 3.0,
            "unit": "x",
        },
    ),
    (
        "Fixed charge coverage at least 2.5x.",
        {
            "metric": "COVERAGE",
            "subtype": "FIXED_CHARGE",
            "op": ">=",
            "value": 2.5,
            "unit": "x",
        },
    ),
    (
        "Debt service coverage ratio shall be between 1.25x and 1.50x.",
        {
            "metric": "COVERAGE",
            "subtype": "DSCR",
            "op": "BETWEEN",
            "min": 1.25,
            "max": 1.50,
            "unit": "x",
        },
    ),
    (
        "DSCR must be between 1.1:1 and 1.4:1.",
        {
            "metric": "COVERAGE",
            "subtype": "DSCR",
            "op": "BETWEEN",
            "min": 1.1,
            "max": 1.4,
            "unit": "x",
        },
    ),
    (
        "Interest coverage minimum 2.0 to 1.",
        {
            "metric": "COVERAGE",
            "subtype": "INTEREST",
            "op": ">=",
            "value": 2.0,
            "unit": "x",
        },
    ),
    (
        "Minimum liquidity of $500 million.",
        {
            "metric": "LIQUIDITY_MIN",
            "op": ">=",
            "value": 500_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Liquidity shall not be less than $250,000,000.",
        {
            "metric": "LIQUIDITY_MIN",
            "op": ">=",
            "value": 250_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Cash and cash equivalents must be at least $75mm.",
        {
            "metric": "LIQUIDITY_MIN",
            "op": ">=",
            "value": 75_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Cash balance minimum of $0.5bn.",
        {
            "metric": "LIQUIDITY_MIN",
            "op": ">=",
            "value": 500_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Minimum liquidity of $300 m.",
        {
            "metric": "LIQUIDITY_MIN",
            "op": ">=",
            "value": 300_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Capex shall not exceed $300mm.",
        {
            "metric": "CAPEX_MAX",
            "op": "<=",
            "value": 300_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Capital expenditures will not exceed $150,000,000.",
        {
            "metric": "CAPEX_MAX",
            "op": "<=",
            "value": 150_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Investments shall not exceed $50 million.",
        {
            "metric": "INVESTMENTS_MAX",
            "op": "<=",
            "value": 50_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Capital expenditure cap at $125m.",
        {
            "metric": "CAPEX_MAX",
            "op": "<=",
            "value": 125_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Capex maximum of $90,000,000.",
        {
            "metric": "CAPEX_MAX",
            "op": "<=",
            "value": 90_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Restricted payments capped at the greater of (i) $100mm and (ii) 50% of Consolidated Net Income.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "GREATER_OF",
            "legs": [
                {"value": 100_000_000.0, "unit": "USD"},
                {"value": 0.5, "unit": "PCT", "basis": "CONSOLIDATED NET INCOME"},
            ],
        },
    ),
    (
        "Dividends limited to the lesser of 2.0x and 60% of EBITDA.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "subtype": "DIVIDENDS",
            "op": "LESSER_OF",
            "legs": [
                {"value": 2.0, "unit": "x"},
                {"value": 0.60, "unit": "PCT", "basis": "EBITDA"},
            ],
        },
    ),
    (
        "Restricted payment capacity shall not exceed $200,000,000.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "<=",
            "value": 200_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Restricted payments no more than $150mm.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "<=",
            "value": 150_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Restricted payments maximum of 65% of CNI.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "<=",
            "value": 0.65,
            "unit": "PCT",
        },
    ),
    (
        "Tangible net worth shall not be less than $400 million.",
        {
            "metric": "TANGIBLE_NET_WORTH",
            "op": ">=",
            "value": 400_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Net working capital minimum of $250mm.",
        {
            "metric": "NET_WORKING_CAPITAL",
            "op": ">=",
            "value": 250_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Current ratio shall not be less than 1.1x.",
        {
            "metric": "CURRENT_RATIO",
            "op": ">=",
            "value": 1.1,
            "unit": "x",
        },
    ),
    (
        "Quick ratio must be at least 1.0x.",
        {
            "metric": "QUICK_RATIO",
            "op": ">=",
            "value": 1.0,
            "unit": "x",
        },
    ),
    (
        "Asset coverage shall not be less than 1.5x.",
        {
            "metric": "ASSET_COVERAGE",
            "op": ">=",
            "value": 1.5,
            "unit": "x",
        },
    ),
    (
        "Loan-to-value ratio shall not exceed 65%.",
        {
            "metric": "LTV",
            "op": "<=",
            "value": 0.65,
            "unit": "PCT",
        },
    ),
    (
        "On a pro forma TTM basis, consolidated net leverage shall not exceed 4.5x.",
        {
            "metric": "LEVERAGE",
            "subtype": "NET",
            "op": "<=",
            "value": 4.5,
            "unit": "x",
            "qualifiers": {
                "pro_forma": True,
                "ttm": True,
                "consolidated": True,
                "net": True,
            },
        },
    ),
    (
        "On a pro forma basis senior secured leverage shall not exceed 3.5x.",
        {
            "metric": "LEVERAGE",
            "subtype": "SENIOR_SECURED",
            "op": "<=",
            "value": 3.5,
            "unit": "x",
            "qualifiers": {"pro_forma": True, "secured": "SENIOR"},
        },
    ),
    (
        "First lien leverage on a consolidated basis shall not exceed 4.0x.",
        {
            "metric": "LEVERAGE",
            "subtype": "FIRST_LIEN",
            "op": "<=",
            "value": 4.0,
            "unit": "x",
            "qualifiers": {"consolidated": True, "secured": "FIRST_LIEN"},
        },
    ),
    (
        "Senior secured net leverage shall not exceed 3.25x.",
        {
            "metric": "LEVERAGE",
            "subtype": "SENIOR_SECURED",
            "op": "<=",
            "value": 3.25,
            "unit": "x",
            "qualifiers": {"secured": "SENIOR", "net": True},
        },
    ),
    (
        "Minimum liquidity of $250\u00a0000\u00a0000 on a consolidated basis.",
        {
            "metric": "LIQUIDITY_MIN",
            "op": ">=",
            "value": 250_000_000.0,
            "unit": "USD",
            "qualifiers": {"consolidated": True},
        },
    ),
    (
        "Capex shall not exceed $275,000,000 in any fiscal year.",
        {
            "metric": "CAPEX_MAX",
            "op": "<=",
            "value": 275_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Investments capped at the greater of $25mm and 25% of EBITDA.",
        {
            "metric": "INVESTMENTS_MAX",
            "op": "GREATER_OF",
            "legs": [
                {"value": 25_000_000.0, "unit": "USD"},
                {"value": 0.25, "unit": "PCT", "basis": "EBITDA"},
            ],
        },
    ),
    (
        "Restricted payments limited to the greater of $75mm and $25 million.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "GREATER_OF",
            "legs": [
                {"value": 75_000_000.0, "unit": "USD"},
                {"value": 25_000_000.0, "unit": "USD"},
            ],
        },
    ),
    (
        "Restricted payments limited to the lesser of 1.5x and 40% of Consolidated Net Income.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "LESSER_OF",
            "legs": [
                {"value": 1.5, "unit": "x"},
                {"value": 0.40, "unit": "PCT", "basis": "CONSOLIDATED NET INCOME"},
            ],
        },
    ),
    (
        "Tangible net worth shall be between $300mm and $350mm.",
        {
            "metric": "TANGIBLE_NET_WORTH",
            "op": "BETWEEN",
            "min": 300_000_000.0,
            "max": 350_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Net working capital shall be between $200 million and $260 million.",
        {
            "metric": "NET_WORKING_CAPITAL",
            "op": "BETWEEN",
            "min": 200_000_000.0,
            "max": 260_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Asset coverage must be at least 1.75x.",
        {
            "metric": "ASSET_COVERAGE",
            "op": ">=",
            "value": 1.75,
            "unit": "x",
        },
    ),
    (
        "Loan to value not to exceed 60%.",
        {
            "metric": "LTV",
            "op": "<=",
            "value": 0.60,
            "unit": "PCT",
        },
    ),
    (
        "Interest coverage shall not be less than 250 bps.",
        {
            "metric": "COVERAGE",
            "subtype": "INTEREST",
            "op": ">=",
            "value": 0.025,
            "unit": "PCT",
        },
    ),
    (
        "Quick ratio shall be between 1.05x and 1.35x.",
        {
            "metric": "QUICK_RATIO",
            "op": "BETWEEN",
            "min": 1.05,
            "max": 1.35,
            "unit": "x",
        },
    ),
    (
        "Current ratio between 1.20x and 1.40x.",
        {
            "metric": "CURRENT_RATIO",
            "op": "BETWEEN",
            "min": 1.20,
            "max": 1.40,
            "unit": "x",
        },
    ),
    (
        "Senior secured leverage between 3.0x and 3.5x.",
        {
            "metric": "LEVERAGE",
            "subtype": "SENIOR_SECURED",
            "op": "BETWEEN",
            "min": 3.0,
            "max": 3.5,
            "unit": "x",
        },
    ),
    (
        "Net leverage between 4.0x and 4.5x on a pro forma basis.",
        {
            "metric": "LEVERAGE",
            "subtype": "NET",
            "op": "BETWEEN",
            "min": 4.0,
            "max": 4.5,
            "unit": "x",
            "qualifiers": {"pro_forma": True, "net": True},
        },
    ),
    (
        "Consolidated net leverage minimum 3.75x.",
        {
            "metric": "LEVERAGE",
            "subtype": "NET",
            "op": ">=",
            "value": 3.75,
            "unit": "x",
            "qualifiers": {"consolidated": True, "net": True},
        },
    ),
    (
        "Minimum consolidated liquidity of $125 million.",
        {
            "metric": "LIQUIDITY_MIN",
            "op": ">=",
            "value": 125_000_000.0,
            "unit": "USD",
            "qualifiers": {"consolidated": True},
        },
    ),
    (
        "Capex maximum of $60,000,000 per year.",
        {
            "metric": "CAPEX_MAX",
            "op": "<=",
            "value": 60_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Investments shall be between $15 million and $25 million.",
        {
            "metric": "INVESTMENTS_MAX",
            "op": "BETWEEN",
            "min": 15_000_000.0,
            "max": 25_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Restricted payments capped at $40mm.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "<=",
            "value": 40_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Restricted payments capped at 30% of Consolidated Net Income.",
        {
            "metric": "RESTRICTED_PAYMENTS",
            "op": "<=",
            "value": 0.30,
            "unit": "PCT",
        },
    ),
    (
        "Tangible net worth shall not be less than $275,000,000 at any time.",
        {
            "metric": "TANGIBLE_NET_WORTH",
            "op": ">=",
            "value": 275_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Net working capital shall not exceed $500mm.",
        {
            "metric": "NET_WORKING_CAPITAL",
            "op": "<=",
            "value": 500_000_000.0,
            "unit": "USD",
        },
    ),
    (
        "Asset coverage shall not exceed 70%.",
        {
            "metric": "ASSET_COVERAGE",
            "op": "<=",
            "value": 0.70,
            "unit": "PCT",
        },
    ),
    (
        "Loan-to-value may not exceed 55%.",
        {
            "metric": "LTV",
            "op": "<=",
            "value": 0.55,
            "unit": "PCT",
        },
    ),
    (
        "Current ratio shall not exceed 2.0x.",
        {
            "metric": "CURRENT_RATIO",
            "op": "<=",
            "value": 2.0,
            "unit": "x",
        },
    ),
    (
        "Quick ratio shall not exceed 1.8x.",
        {
            "metric": "QUICK_RATIO",
            "op": "<=",
            "value": 1.8,
            "unit": "x",
        },
    ),
    (
        "Tangible net worth greater of $150mm and 50% of EBITDA.",
        {
            "metric": "TANGIBLE_NET_WORTH",
            "op": "GREATER_OF",
            "legs": [
                {"value": 150_000_000.0, "unit": "USD"},
                {"value": 0.50, "unit": "PCT", "basis": "EBITDA"},
            ],
        },
    ),
    (
        "Net leverage lesser of 4.0x and 45% of EBITDA.",
        {
            "metric": "LEVERAGE",
            "subtype": "NET",
            "op": "LESSER_OF",
            "legs": [
                {"value": 4.0, "unit": "x"},
                {"value": 0.45, "unit": "PCT", "basis": "EBITDA"},
            ],
        },
    ),
]

assert len(BASIC_CASES) >= 60


@pytest.mark.parametrize("text, expected", BASIC_CASES)
def test_numeric_extractor_cases(text, expected):
    result = _expect_single(text)
    assert result["metric"] == expected.get("metric")
    assert result.get("subtype") == expected.get("subtype")
    assert result["op"] == expected.get("op")

    if "value" in expected:
        assert result.get("value") == pytest.approx(expected["value"])
    if "min" in expected:
        assert result.get("min") == pytest.approx(expected["min"])
    if "max" in expected:
        assert result.get("max") == pytest.approx(expected["max"])
    if "unit" in expected:
        assert result.get("unit") == expected["unit"]
    if "legs" in expected:
        assert result.get("legs") is not None
        actual_legs = result["legs"]
        assert len(actual_legs) == len(expected["legs"])
        for actual, expected_leg in zip(actual_legs, expected["legs"]):
            assert actual["unit"] == expected_leg["unit"]
            assert actual.get("value") == pytest.approx(expected_leg["value"])
            if "basis" in expected_leg:
                assert actual.get("basis") == expected_leg["basis"]
    if "qualifiers" in expected:
        for key, value in expected["qualifiers"].items():
            assert result["qualifiers"].get(key) == value

    assert result["confidence"] >= 0.6


FALSE_POSITIVES = [
    "Revenue grew 5% year-over-year.",
    "Note 5. Property and Equipment.",
    "The company had cash of $400 million at year end.",
    "Operating income increased 10% year over year.",
    "Management discussed leverage trends without hard targets.",
]


@pytest.mark.parametrize("text", FALSE_POSITIVES)
def test_numeric_extractor_false_positives(text):
    assert extract_numeric_covenants(text) == []
