import pytest

from potatobacon.tariff.batch_scan import batch_scan_tariffs
from potatobacon.tariff.models import TariffBatchScanRequestModel, TariffBatchSkuModel


def _build_request(include_random: bool = False):
    skus = [
        TariffBatchSkuModel(
            sku_id="CONVERSE_DEMO",
            description="Canvas sneaker with rubber sole",
            declared_value_per_unit=100.0,
            annual_volume=10_000,
        ),
        TariffBatchSkuModel(
            sku_id="TESLA_BOLT_X1",
            description="Chassis bolt fastener for EV frame",
            declared_value_per_unit=200.0,
            annual_volume=50_000,
        ),
    ]
    if include_random:
        skus.append(
            TariffBatchSkuModel(
                sku_id="RANDOM_GADGET",
                description="random gadget",
                declared_value_per_unit=50.0,
            )
        )

    return TariffBatchScanRequestModel(
        skus=skus,
        top_k_per_sku=3,
        max_results=10,
        seed=2025,
    )


def test_ranks_tesla_above_converse():
    request = _build_request()
    response = batch_scan_tariffs(request)

    assert response.status == "OK"
    assert [result.sku_id for result in response.results[:2]] == [
        "TESLA_BOLT_X1",
        "CONVERSE_DEMO",
    ]

    tesla = response.results[0]
    converse = response.results[1]

    assert pytest.approx(tesla.best.annual_savings_value, rel=1e-6) == 400_000.0
    assert pytest.approx(tesla.best.optimized_duty_rate, rel=1e-6) == 2.5
    assert tesla.best.proof_id

    assert pytest.approx(converse.best.annual_savings_value, rel=1e-6) == 345_000.0
    assert pytest.approx(converse.best.optimized_duty_rate, rel=1e-6) == 3.0
    assert converse.best.proof_id


def test_unknown_sku_goes_to_skipped():
    request = _build_request(include_random=True)
    response = batch_scan_tariffs(request)

    assert len(response.results) == 2
    assert any(item.sku_id == "RANDOM_GADGET" for item in response.skipped)

    skipped_item = next(item for item in response.skipped if item.sku_id == "RANDOM_GADGET")
    assert skipped_item.status == "NO_CANDIDATES"
    assert skipped_item.best is None


def test_batch_scan_is_deterministic():
    request = _build_request(include_random=True)

    first = batch_scan_tariffs(request)
    second = batch_scan_tariffs(request)

    assert [r.sku_id for r in first.results] == [r.sku_id for r in second.results]

    for res_a, res_b in zip(first.results, second.results):
        assert res_a.rank_score == res_b.rank_score
        assert res_a.best.optimized_duty_rate == res_b.best.optimized_duty_rate
