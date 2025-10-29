"""End-to-end repository audit for the offline CALE event study."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - defensive import path tweak
    sys.path.insert(0, str(ROOT))

import tools.event_study_core as core
import tools.event_study_delta as event_study_delta


REPORT_ROOT = Path("reports/audit")
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def _format_float(value: float) -> str:
    if value != value:  # NaN check
        return "nan"
    return f"{value:.3f}"


def _render_report(metrics: Dict[str, object], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = report_dir / "REPORT.md"
    baseline = metrics.get("baseline", {})
    logistic = metrics.get("logistic", {})
    density = metrics.get("evidence_density", {})
    false_pos = metrics.get("false_positives_ig", {})
    verdict = metrics.get("pass_fail", {}).get("verdict", "NOT_READY")

    lines = ["# CALE Event Study Audit", "", f"Verdict: **{verdict}**", ""]
    lines.append("## Metrics")
    lines.append(f"- Baseline AUC: {_format_float(baseline.get('auc', float('nan')))}")
    lines.append(f"- Logistic AUC: {_format_float(logistic.get('auc', float('nan')))}")
    lines.append(f"- Baseline p-value: {_format_float(baseline.get('p_value', float('nan')))}")
    fp_rate = false_pos.get("rate")
    lines.append(f"- IG False-positive rate: {_format_float(fp_rate if fp_rate is not None else float('nan'))}")
    lines.append(
        "- Avg evidence pairs (distressed): "
        f"{_format_float(density.get('avg_pairs_distressed', float('nan')))}"
    )
    lines.append(
        "- Avg evidence pairs (control): "
        f"{_format_float(density.get('avg_pairs_control', float('nan')))}"
    )
    lines.append(f"- Min pairs (distressed): {_format_float(density.get('min_pairs_distressed', float('nan')))}")
    lines.append("")
    lines.append("## Checklist")
    for key in ("auc", "p_value", "fp_rate", "pair_density"):
        status = "PASS" if metrics.get("pass_fail", {}).get(key, False) else "FAIL"
        lines.append(f"- {key}: {status}")
    lines.append("")
    lines.append("Metrics payload saved to `reports/leverage_alpha/metrics.json`.")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    (report_dir / "REPORT.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    core.configure_cli(parser)
    args = parser.parse_args()

    if Path("reports/leverage_alpha/metrics.json").exists():
        # Rebuild using supplied CLI to keep determinism.
        metrics = core.run_event_study(args)
    else:
        metrics = core.run_event_study(args)

    event_study_delta.run_delta(args)

    report_dir = REPORT_ROOT / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    _render_report(metrics, report_dir)

    verdict = metrics.get("pass_fail", {}).get("verdict", "NOT_READY")
    print(f"Executive summary: {verdict}")
    print(f"Report written to {report_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

