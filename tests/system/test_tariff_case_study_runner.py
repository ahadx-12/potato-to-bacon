from pathlib import Path

from fastapi.testclient import TestClient

from potatobacon.tariff.case_study import run_case_study


def test_tariff_case_study_runner(system_client: TestClient, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports" / "case_studies"
    result = run_case_study(
        system_client,
        seed=2025,
        output_dir=output_dir,
        timestamp="20250101T000000Z",
    )

    baseline = result["baseline"]
    refine = result["refine"]

    assert baseline["status"] != "ERROR"
    assert refine["status"] in {"OK_OPTIMIZED", "OK_CONDITIONAL_OPTIMIZATION"}
    assert refine["net_savings"]["net_annual_savings"] > 0
    assert baseline["proof_replay"]["ok"] is True
    assert refine["proof_replay"]["ok"] is True

    audit_pack = Path(result["audit_pack"]["path"])
    assert audit_pack.exists()
    assert audit_pack.read_bytes()
