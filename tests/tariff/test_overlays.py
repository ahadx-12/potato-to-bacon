from potatobacon.tariff.overlays import effective_duty_rate, evaluate_overlays


def test_section301_overlay_applies_for_cn_prefix():
    overlays = evaluate_overlays(
        facts={"origin_country_CN": True, "hts_code": "6404.19"},
        active_codes=["HTS_6404_11_90"],
        origin_country="CN",
        import_country="US",
        data_root="data/overlays",
    )

    assert overlays, "Expected Section 301 overlay to apply for CN origin and 6404 code"
    overlay_names = [ov.overlay_name for ov in overlays]
    assert any("Section 301" in name for name in overlay_names)
    assert overlays[0].additional_rate == 25.0
    assert overlays[0].stop_optimization is True
    assert effective_duty_rate(3.0, overlays) == 28.0
