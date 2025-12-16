import pytest

from potatobacon.tariff.models import TariffSuggestRequestModel
from potatobacon.tariff.suggest import suggest_tariff_optimizations


@pytest.mark.parametrize(
    "description, expected_baseline, expected_optimized",
    [
        ("canvas sneaker with rubber sole", 37.5, 3.0),
        ("chassis bolt fastener for EV frame", 6.5, 2.5),
    ],
)
def test_suggest_engine_generates_expected_rates(description, expected_baseline, expected_optimized):
    request = TariffSuggestRequestModel(
        description=description,
        declared_value_per_unit=100.0,
        annual_volume=10_000,
    )

    response = suggest_tariff_optimizations(request)

    assert response.status == "OK"
    assert response.suggestions
    top = response.suggestions[0]
    assert pytest.approx(top.baseline_duty_rate, rel=1e-6) == expected_baseline
    assert pytest.approx(top.optimized_duty_rate, rel=1e-6) == expected_optimized
    assert isinstance(top.proof_id, str) and top.proof_id


def test_suggest_engine_handles_unknown_category():
    request = TariffSuggestRequestModel(description="random gadget with no match")

    response = suggest_tariff_optimizations(request)

    assert response.status == "NO_CANDIDATES"
    assert response.suggestions == []
