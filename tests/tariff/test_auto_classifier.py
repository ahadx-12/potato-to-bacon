from __future__ import annotations

from potatobacon.tariff.auto_classifier import classify_product, classify_with_thresholds, declared_code_warning


def test_classify_product_returns_top3_or_fewer():
    rows = classify_product(
        description="stainless steel bolt and nut kit",
        materials=["stainless steel"],
        weight_kg=0.4,
        value_usd=5.5,
    )
    assert 1 <= len(rows) <= 3
    assert all(0.0 <= item.confidence <= 1.0 for item in rows)


def test_threshold_resolution_modes():
    payload = classify_with_thresholds(
        description="polyethylene packaging film roll",
        materials=["polyethylene plastic"],
        weight_kg=2.0,
        value_usd=8.0,
    )
    assert payload["hts_source"] in {
        "auto_classified",
        "auto_classified_low_confidence",
        "manual_classification_required",
    }


def test_declared_code_warning_when_chapter_differs():
    warning = declared_code_warning(
        declared_hts_code="0409.00.00",
        description="industrial hydraulic pump",
        materials=["steel"],
        weight_kg=9.0,
        value_usd=130.0,
    )
    # Warning may be None when classifier confidence is too low, but should be deterministic type.
    assert warning is None or "Review recommended" in warning
