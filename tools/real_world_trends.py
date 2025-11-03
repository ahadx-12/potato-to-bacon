#!/usr/bin/env python3
"""Aggregate real-world validation runs into Markdown and HTML dashboards."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HISTORY_CSV = Path("reports/realworld/history.csv")
DASHBOARD_MD = Path("reports/realworld/dashboard.md")
DASHBOARD_HTML = Path("reports/realworld/dashboard.html")
RUNS_DIR = Path("reports/realworld/runs")


@dataclass
class RunEntry:
    run_id: str
    timestamp: datetime
    timestamp_str: str
    git_sha: str
    baseline_auc: float
    baseline_ci: Tuple[float, float]
    logistic_auc: float
    logistic_ci: Tuple[float, float]
    delta_auc: float
    p_value: float
    p_ci: Tuple[float, float]
    ig_fp_rate: float
    ig_fp_count: float
    ig_fp_total: float
    avg_pairs_distressed: float
    avg_pairs_control: float
    avg_pairs_overall: float
    records_total: int


def _float_or_nan(value: object) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_zero(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _ci_pair(row: Dict[str, object], low_key: str, high_key: str) -> Tuple[float, float]:
    return _float_or_nan(row.get(low_key)), _float_or_nan(row.get(high_key))


def _load_history_from_csv() -> List[RunEntry]:
    if not HISTORY_CSV.exists():
        return []
    entries: List[RunEntry] = []
    with HISTORY_CSV.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp_str = row.get("timestamp", "")
            entries.append(
                RunEntry(
                    run_id=row.get("run_id", "unknown"),
                    timestamp=_parse_timestamp(timestamp_str),
                    timestamp_str=timestamp_str,
                    git_sha=row.get("git_sha", "unknown"),
                    baseline_auc=_float_or_nan(row.get("baseline_auc")),
                    baseline_ci=_ci_pair(row, "baseline_auc_ci_low", "baseline_auc_ci_high"),
                    logistic_auc=_float_or_nan(row.get("logistic_auc")),
                    logistic_ci=_ci_pair(row, "logistic_auc_ci_low", "logistic_auc_ci_high"),
                    delta_auc=_float_or_nan(row.get("delta_auc")),
                    p_value=_float_or_nan(row.get("welch_p_value")),
                    p_ci=_ci_pair(row, "welch_p_ci_low", "welch_p_ci_high"),
                    ig_fp_rate=_float_or_nan(row.get("ig_fp_rate")),
                    ig_fp_count=_float_or_nan(row.get("ig_fp_count")),
                    ig_fp_total=_float_or_nan(row.get("ig_fp_total")),
                    avg_pairs_distressed=_float_or_nan(row.get("avg_pairs_distressed")),
                    avg_pairs_control=_float_or_nan(row.get("avg_pairs_control")),
                    avg_pairs_overall=_float_or_nan(row.get("avg_pairs_overall")),
                    records_total=_int_or_zero(row.get("records_total")),
                )
            )
    return entries


def _load_history_from_runs() -> List[RunEntry]:
    if not RUNS_DIR.exists():
        return []
    entries: List[RunEntry] = []
    for path in sorted(RUNS_DIR.glob("*_metrics.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summary = data.get("summary", {})
        if not summary:
            continue
        timestamp = summary.get("timestamp") or data.get("timestamp") or ""
        entries.append(
            RunEntry(
                run_id=summary.get("run_id", path.stem),
                timestamp=_parse_timestamp(timestamp),
                timestamp_str=timestamp,
                git_sha=summary.get("git_sha", data.get("git_sha", "unknown")),
                baseline_auc=_float_or_nan(summary.get("baseline_auc")),
                baseline_ci=(
                    _float_or_nan(summary.get("baseline_auc_ci_low")),
                    _float_or_nan(summary.get("baseline_auc_ci_high")),
                ),
                logistic_auc=_float_or_nan(summary.get("logistic_auc")),
                logistic_ci=(
                    _float_or_nan(summary.get("logistic_auc_ci_low")),
                    _float_or_nan(summary.get("logistic_auc_ci_high")),
                ),
                delta_auc=_float_or_nan(summary.get("delta_auc")),
                p_value=_float_or_nan(summary.get("welch_p_value")),
                p_ci=(
                    _float_or_nan(summary.get("welch_p_ci_low")),
                    _float_or_nan(summary.get("welch_p_ci_high")),
                ),
                ig_fp_rate=_float_or_nan(summary.get("ig_fp_rate")),
                ig_fp_count=_float_or_nan(summary.get("ig_fp_count")),
                ig_fp_total=_float_or_nan(summary.get("ig_fp_total")),
                avg_pairs_distressed=_float_or_nan(summary.get("avg_pairs_distressed")),
                avg_pairs_control=_float_or_nan(summary.get("avg_pairs_control")),
                avg_pairs_overall=_float_or_nan(summary.get("avg_pairs_overall")),
                records_total=_int_or_zero(summary.get("records_total")),
            )
        )
    return entries


def _load_history() -> List[RunEntry]:
    entries = _load_history_from_csv()
    if entries:
        return sorted(entries, key=lambda entry: entry.timestamp)
    entries = _load_history_from_runs()
    return sorted(entries, key=lambda entry: entry.timestamp)


def _format_float(value: float, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_percent(value: float) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_ci(ci: Tuple[float, float]) -> str:
    low, high = ci
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"{low:.3f}–{high:.3f}"


def _sparkline(values: Sequence[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return "n/a"
    clean = [val for val in values if not math.isnan(val)]
    if not clean:
        return "n/a"
    minimum = min(clean)
    maximum = max(clean)
    if math.isclose(minimum, maximum):
        return blocks[len(blocks) // 2] * len(values)
    span = maximum - minimum
    spark_chars: List[str] = []
    for val in values:
        if math.isnan(val):
            spark_chars.append(" ")
            continue
        scaled = (val - minimum) / span
        idx = min(len(blocks) - 1, max(0, int(round(scaled * (len(blocks) - 1)))))
        spark_chars.append(blocks[idx])
    return "".join(spark_chars)


def _metric_summary(values: Sequence[float]) -> Dict[str, object]:
    clean = [val for val in values if not math.isnan(val)]
    return {
        "spark": _sparkline(values),
        "min": min(clean) if clean else float("nan"),
        "max": max(clean) if clean else float("nan"),
        "values": values,
    }


def _format_values(values: Sequence[float]) -> str:
    return ", ".join(_format_float(val) for val in values if not math.isnan(val)) or "n/a"


def _build_markdown(entries: Sequence[RunEntry]) -> str:
    generated = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    lines: List[str] = []
    lines.append("# Real-World Validation Trends")
    lines.append(f"_Generated_: {generated}")
    lines.append("")
    if not entries:
        lines.append("No completed validation runs found. Execute tools/real_world_validate.py to seed history.")
        return "\n".join(lines)

    lines.append("## Run summary")
    lines.append("| # | Timestamp | Git SHA | Records | Baseline AUC | 95% CI | Welch p-value | 95% CI | Logistic AUC | 95% CI | IG FP rate | Evidence density |")
    lines.append("|---|-----------|---------|---------|--------------|-------|---------------|-------|--------------|-------|-----------|-----------------|")
    for idx, entry in enumerate(entries, start=1):
        ig_display = f"{_format_percent(entry.ig_fp_rate)} ({int(entry.ig_fp_count)}/{int(entry.ig_fp_total)})"
        density_display = (
            f"dist {_format_float(entry.avg_pairs_distressed)} | "
            f"ctrl {_format_float(entry.avg_pairs_control)} | avg {_format_float(entry.avg_pairs_overall)}"
        )
        lines.append(
            "| {idx} | {ts} | {sha} | {records} | {auc} | {auc_ci} | {p} | {p_ci} | {log_auc} | {log_ci} | {ig} | {density} |".format(
                idx=idx,
                ts=entry.timestamp_str or "n/a",
                sha=entry.git_sha,
                records=entry.records_total,
                auc=_format_float(entry.baseline_auc),
                auc_ci=_format_ci(entry.baseline_ci),
                p=_format_float(entry.p_value),
                p_ci=_format_ci(entry.p_ci),
                log_auc=_format_float(entry.logistic_auc),
                log_ci=_format_ci(entry.logistic_ci),
                ig=ig_display,
                density=density_display,
            )
        )

    baseline_summary = _metric_summary([entry.baseline_auc for entry in entries])
    logistic_summary = _metric_summary([entry.logistic_auc for entry in entries])
    delta_summary = _metric_summary([entry.delta_auc for entry in entries])
    ig_summary = _metric_summary([entry.ig_fp_rate for entry in entries])
    dist_density_summary = _metric_summary([entry.avg_pairs_distressed for entry in entries])
    ctrl_density_summary = _metric_summary([entry.avg_pairs_control for entry in entries])
    overall_density_summary = _metric_summary([entry.avg_pairs_overall for entry in entries])

    lines.append("")
    lines.append("## Trajectories")
    lines.append("### Baseline AUC")
    lines.append(f"`{baseline_summary['spark']}` (min {_format_float(baseline_summary['min'])}, max {_format_float(baseline_summary['max'])})")
    lines.append(f"Values: {_format_values(baseline_summary['values'])}")
    lines.append("")
    lines.append("### Logistic AUC")
    lines.append(f"`{logistic_summary['spark']}` (min {_format_float(logistic_summary['min'])}, max {_format_float(logistic_summary['max'])})")
    lines.append(f"Values: {_format_values(logistic_summary['values'])}")
    lines.append("")
    lines.append("### ΔCCE AUC")
    lines.append(f"`{delta_summary['spark']}` (min {_format_float(delta_summary['min'])}, max {_format_float(delta_summary['max'])})")
    lines.append(f"Values: {_format_values(delta_summary['values'])}")
    lines.append("")
    lines.append("### IG false-positive rate")
    lines.append(f"`{ig_summary['spark']}` (min {_format_float(ig_summary['min'])}, max {_format_float(ig_summary['max'])})")
    lines.append(f"Values: {_format_values(ig_summary['values'])}")
    lines.append("")
    lines.append("### Evidence density — distressed")
    lines.append(f"`{dist_density_summary['spark']}` (min {_format_float(dist_density_summary['min'])}, max {_format_float(dist_density_summary['max'])})")
    lines.append(f"Values: {_format_values(dist_density_summary['values'])}")
    lines.append("")
    lines.append("### Evidence density — control")
    lines.append(f"`{ctrl_density_summary['spark']}` (min {_format_float(ctrl_density_summary['min'])}, max {_format_float(ctrl_density_summary['max'])})")
    lines.append(f"Values: {_format_values(ctrl_density_summary['values'])}")
    lines.append("")
    lines.append("### Evidence density — overall avg")
    lines.append(f"`{overall_density_summary['spark']}` (min {_format_float(overall_density_summary['min'])}, max {_format_float(overall_density_summary['max'])})")
    lines.append(f"Values: {_format_values(overall_density_summary['values'])}")

    return "\n".join(lines)


def _build_html(entries: Sequence[RunEntry]) -> str:
    generated = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    html: List[str] = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang=\"en\">")
    html.append("<head>")
    html.append("  <meta charset=\"utf-8\" />")
    html.append("  <title>Real-World Validation Trends</title>")
    html.append("  <style>body{font-family:system-ui,sans-serif;margin:2rem;} table{border-collapse:collapse;margin-bottom:1.5rem;} th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;} code.spark{font-family:'Fira Code',monospace,monospace;font-size:1.1rem;display:inline-block;margin:0.25rem 0;} .muted{color:#555;} </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("  <h1>Real-World Validation Trends</h1>")
    html.append(f"  <p class=\"muted\"><em>Generated: {generated}</em></p>")

    if not entries:
        html.append("  <p>No completed validation runs found. Execute <code>tools/real_world_validate.py</code> to seed history.</p>")
        html.append("</body></html>")
        return "\n".join(html)

    html.append("  <h2>Run summary</h2>")
    html.append("  <table>")
    html.append("    <thead><tr><th>#</th><th>Timestamp</th><th>Git SHA</th><th>Records</th><th>Baseline AUC</th><th>95% CI</th><th>Welch p-value</th><th>95% CI</th><th>Logistic AUC</th><th>95% CI</th><th>IG FP rate</th><th>Evidence density</th></tr></thead>")
    html.append("    <tbody>")
    for idx, entry in enumerate(entries, start=1):
        ig_display = f"{_format_percent(entry.ig_fp_rate)} ({int(entry.ig_fp_count)}/{int(entry.ig_fp_total)})"
        density_display = (
            f"dist {_format_float(entry.avg_pairs_distressed)} | "
            f"ctrl {_format_float(entry.avg_pairs_control)} | avg {_format_float(entry.avg_pairs_overall)}"
        )
        html.append(
            "      <tr><td>{idx}</td><td>{ts}</td><td>{sha}</td><td>{records}</td><td>{auc}</td><td>{auc_ci}</td><td>{p}</td><td>{p_ci}</td><td>{log_auc}</td><td>{log_ci}</td><td>{ig}</td><td>{density}</td></tr>".format(
                idx=idx,
                ts=entry.timestamp_str or "n/a",
                sha=entry.git_sha,
                records=entry.records_total,
                auc=_format_float(entry.baseline_auc),
                auc_ci=_format_ci(entry.baseline_ci),
                p=_format_float(entry.p_value),
                p_ci=_format_ci(entry.p_ci),
                log_auc=_format_float(entry.logistic_auc),
                log_ci=_format_ci(entry.logistic_ci),
                ig=ig_display,
                density=density_display,
            )
        )
    html.append("    </tbody>")
    html.append("  </table>")

    metric_blocks = [
        ("Baseline AUC", _metric_summary([entry.baseline_auc for entry in entries])),
        ("Logistic AUC", _metric_summary([entry.logistic_auc for entry in entries])),
        ("ΔCCE AUC", _metric_summary([entry.delta_auc for entry in entries])),
        ("IG false-positive rate", _metric_summary([entry.ig_fp_rate for entry in entries])),
        ("Evidence density — distressed", _metric_summary([entry.avg_pairs_distressed for entry in entries])),
        ("Evidence density — control", _metric_summary([entry.avg_pairs_control for entry in entries])),
        ("Evidence density — overall avg", _metric_summary([entry.avg_pairs_overall for entry in entries])),
    ]

    html.append("  <h2>Trajectories</h2>")
    for title, summary in metric_blocks:
        html.append(f"  <h3>{title}</h3>")
        spark = summary["spark"]
        html.append(f"  <code class=\"spark\">{spark}</code>")
        html.append(
            "  <p class=\"muted\">min {min_val}, max {max_val}</p>".format(
                min_val=_format_float(summary["min"]),
                max_val=_format_float(summary["max"]),
            )
        )
        html.append(f"  <p class=\"muted\">Values: {_format_values(summary['values'])}</p>")

    html.append("</body></html>")
    return "\n".join(html)


def main() -> int:
    entries = _load_history()
    DASHBOARD_MD.parent.mkdir(parents=True, exist_ok=True)
    markdown_content = _build_markdown(entries)
    DASHBOARD_MD.write_text(markdown_content.rstrip() + "\n", encoding="utf-8")
    html_content = _build_html(entries)
    if not html_content.endswith("\n"):
        html_content += "\n"
    DASHBOARD_HTML.write_text(html_content, encoding="utf-8")
    print(DASHBOARD_MD.resolve())
    print(DASHBOARD_HTML.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
