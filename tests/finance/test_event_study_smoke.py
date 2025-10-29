from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tools import event_study_core as core


def test_pandas_available() -> None:
    pd = core.ensure_pandas()
    assert hasattr(pd, "DataFrame")


def test_event_study_pipeline(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    args = SimpleNamespace(
        events_csv=Path("data/events/events.csv"),
        controls_csv=Path("data/events/controls.csv"),
        out_dir=out_dir,
        min_filings=12,
        seed=11,
    )
    metrics = core.run_event_study(args)

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()

    baseline = metrics.get("baseline", {})
    logistic = metrics.get("logistic", {})
    evidence = metrics.get("evidence_density", {})

    assert "auc" in baseline and baseline["auc"] >= 0.5
    assert "auc" in logistic and logistic["auc"] >= 0.5
    assert evidence.get("avg_pairs_distressed", 0.0) >= 1.0

