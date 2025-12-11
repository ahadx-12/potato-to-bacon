from tests.experiments.exp_tariff_converse import run_experiment


def test_converse_felt_overlay_experiment(tmp_path):
    dossier, report_path = run_experiment(output_dir=tmp_path)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Status: 🟢 OPTIMIZED" in content
    assert "Baseline Cost: $37.50" in content
    assert "Optimized Cost: $3.00" in content
    assert "Savings: $34.50" in content

    assert dossier["baseline_duty_rate"] == 37.5
    assert dossier["optimized_duty_rate"] == 3.0
    assert dossier["savings_per_unit"] == 34.5
    assert dossier["baseline_scenario"]["felt_covering_gt_50"] is False
    assert dossier["optimized_scenario"]["felt_covering_gt_50"] is True
    assert dossier["optimized_scenario"]["surface_contact_textile_gt_50"] is True
    assert dossier["optimized_scenario"]["surface_contact_rubber_gt_50"] is False
