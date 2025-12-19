from pathlib import Path

from potatobacon.tariff.e2e_runner import TariffE2ERunner


def test_tariff_e2e_subset(tmp_path):
    runner = TariffE2ERunner(
        dataset_path=Path("tests/data/e2e_pilot_pack.json"),
        mode="engine",
        seed=2025,
        output_path=tmp_path / "report.md",
    )

    result = runner.run(limit=12)

    assert result.determinism.passed, f"Determinism failed: {result.determinism.details}"
    assert abs(result.proof_replay_pass_rate - 1.0) < 1e-9

    ok_results = [res for res in result.sku_results if res.status == "OK" and res.proof_replay]
    assert all(res.proof_replay.ok for res in ok_results)

    assert result.report_path.exists()
    report_text = result.report_path.read_text()
    for section in ["Executive summary", "Category scorecard", "Top opportunities", "Weaknesses"]:
        assert section in report_text
