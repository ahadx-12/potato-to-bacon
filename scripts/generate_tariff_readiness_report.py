from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID
from potatobacon.tariff.readiness_eval import run_readiness_eval


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def _render_improvement(title: str, impact: str, effort: str, dependencies: str) -> str:
    return f"- **{title}** — Impact: {impact}; Effort: {effort}; Dependencies: {dependencies}"


def _build_improvement_plan(aggregates: Dict[str, object]) -> list[str]:
    ok_pct = aggregates.get("ok_pct", 0.0)
    improvements: list[str] = []
    improvements.append(
        _render_improvement(
            "Expand parser coverage for electronics/textile SKUs",
            "High",
            "Medium",
            "Add category keywords + atoms for PCB housings and apparel blends",
        )
    )
    improvements.append(
        _render_improvement(
            "Broaden mutation library beyond footwear/fasteners",
            "High",
            "Medium",
            "Design defensible mutations for electronics enclosures and apparel",
        )
    )
    improvements.append(
        _render_improvement(
            "Integrate origin/FTA & exclusion logic",
            "Medium",
            "High",
            "Requires origin data ingestion + new rule atoms",
        )
    )
    improvements.append(
        _render_improvement(
            "Persist structured BOM ingestion (CSV/JSON)",
            "Medium",
            "Medium",
            "Add upload/parse pipeline and schema validation",
        )
    )
    improvements.append(
        _render_improvement(
            "Add AD/CVD + ruling/precedent integration",
            "High",
            "High",
            "Link to rulings corpus and maintain versioned citations",
        )
    )
    improvements.append(
        _render_improvement(
            "Improve evidence density and snippet extraction",
            "Medium",
            "Low",
            "More robust keyword windows + BOM parsing",
        )
    )
    improvements.append(
        _render_improvement(
            "Strengthen mutation feasibility constraints",
            "Medium",
            "Medium",
            "Capture engineering constraints + cost models",
        )
    )
    improvements.append(
        _render_improvement(
            "Add seed-aware replay CLI and archived manifests",
            "Medium",
            "Low",
            "Reuse proof payload hash + manifest snapshot",
        )
    )
    improvements.append(
        _render_improvement(
            "Expose batch audit metrics via API",
            "Low",
            "Low",
            "Wrap readiness_eval outputs in JSON endpoint",
        )
    )
    if ok_pct < 90:
        improvements.append(
            _render_improvement(
                "Increase category recall for ambiguous gadgets",
                "Medium",
                "Low",
                "Fallback heuristics + ML embeddings for unknown SKUs",
            )
        )
    else:
        improvements.append(
            _render_improvement(
                "Harden rate-limit observability under burst load",
                "Medium",
                "Low",
                "Add structured metrics + alert thresholds",
            )
        )
    return improvements


def _render_category_snapshot(categories: Dict[str, int]) -> str:
    if not categories:
        return "No category data captured."
    lines = []
    for category, metrics in sorted(categories.items()):
        count = metrics.get("processed", 0) if isinstance(metrics, dict) else metrics
        lines.append(f"- {category}: {count} SKUs")
    return "\n".join(lines)


def _render_category_scorecard(by_category: Dict[str, Dict[str, object]]) -> str:
    if not by_category:
        return "No category coverage recorded."
    lines = ["| Category | Processed | OK rate | NO_CANDIDATES | Errors | Avg annual savings | Avg risk | Load-bearing evidence |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for category in sorted(by_category):
        metrics = by_category[category]
        lines.append(
            "| {cat} | {processed} | {ok_rate:.1f}% | {no_cand} | {errors} | {avg_savings} | {avg_risk} | {load_bearing:.1f}% |".format(
                cat=category,
                processed=metrics.get("processed", 0),
                ok_rate=metrics.get("ok_rate", 0.0),
                no_cand=metrics.get("no_candidates", 0),
                errors=metrics.get("errors", 0),
                avg_savings=(
                    f"${metrics['avg_best_annual_savings']:.2f}" if metrics.get("avg_best_annual_savings") is not None else "n/a"
                ),
                avg_risk=f"{metrics['avg_risk_score']:.1f}" if metrics.get("avg_risk_score") is not None else "n/a",
                load_bearing=metrics.get("load_bearing_evidence_rate", 0.0),
            )
        )
    return "\n".join(lines)


def _compute_known_limitations(aggregates: Dict[str, object]) -> list[str]:
    limitations: list[str] = []
    by_category = aggregates.get("by_category", {}) or {}
    if by_category:
        max_no_candidates = max((metrics.get("no_candidates", 0) for metrics in by_category.values()), default=0)
        worst_no_candidates = [
            name for name, metrics in by_category.items() if metrics.get("no_candidates", 0) == max_no_candidates and max_no_candidates > 0
        ]
        if worst_no_candidates:
            limitations.append(f"Highest NO_CANDIDATES: {', '.join(sorted(worst_no_candidates))} ({max_no_candidates})")

    unknown_bucket = by_category.get("other") or {}
    unknown_processed = unknown_bucket.get("processed", 0)
    if unknown_processed:
        limitations.append(f"Unknown classification volume remains {unknown_processed}")

    origin_stats = aggregates.get("origin_stats", {}) or {}
    limitations.append(
        f"Origin missing rate: {100.0 - origin_stats.get('pct_origin_provided', 0.0):.1f}%"
    )
    limitations.append(
        f"AD/CVD possible rate: {origin_stats.get('pct_ad_cvd_possible', 0.0):.1f}%"
    )
    return limitations


def generate_report() -> Path:
    law_context = DEFAULT_CONTEXT_ID
    results = run_readiness_eval(law_context=law_context, top_k=3, include_evidence=True)
    aggregates = results["aggregates"]

    timestamp = datetime.now(timezone.utc).isoformat()
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"CALE-TARIFF_Readiness_Audit_{timestamp}.md"

    ok_pct = aggregates.get("ok_pct", 0.0)
    error_pct = (aggregates.get("errors", 0) / aggregates.get("processed", 1)) * 100.0
    top_savings = aggregates.get("top_savings", [])
    top_line_items = [
        f"{entry['sku_id']}: ${entry['annual_savings_value']:.2f} ({entry.get('summary')})"
        for entry in top_savings
    ]

    risk_distribution = aggregates.get("risk_distribution", {})
    risk_lines = [f"- {grade}: {count}" for grade, count in sorted(risk_distribution.items())]
    evidence = aggregates.get("evidence", {})
    evidence_coverage = aggregates.get("evidence_coverage_rate", 0.0)
    load_bearing_rate = aggregates.get("load_bearing_evidence_rate", 0.0)
    by_category = aggregates.get("by_category", {})
    improvements = _build_improvement_plan(aggregates)
    limitations = _compute_known_limitations(aggregates)

    determinism = results.get("determinism", {})
    determinism_status = "PASS" if determinism.get("passed") else "FAIL"
    origin_stats = aggregates.get("origin_stats", {})

    content = f"""# CALE-TARIFF Readiness Audit — {timestamp}

## Executive summary
- Law context: **{law_context}**
- OK rate: **{ok_pct:.1f}%**, Errors: **{error_pct:.1f}%**
- Top savings SKUs: {', '.join(top_line_items[:3]) if top_line_items else 'n/a'}
- Determinism & proof replay: **{determinism_status}**

## Coverage snapshot
{_render_category_snapshot(by_category or aggregates.get('category_breakdown', {}))}

## Category scorecard
{_render_category_scorecard(by_category)}

## Evidence quality summary
- Facts with evidence: {evidence.get('facts_with_evidence', 0)} / {evidence.get('total_facts', 0)}
- Snippets captured: {evidence.get('snippets', 0)}
- Evidence requested: **True**
- Evidence coverage: {_format_percentage(evidence_coverage)}
- Load-bearing evidence: {_format_percentage(load_bearing_rate)}

## Origin and AD/CVD overview
- Origin provided: {_format_percentage(origin_stats.get('pct_origin_provided'))}
- Requires origin data: {_format_percentage(origin_stats.get('pct_requires_origin_data'))}
- AD/CVD possible: {_format_percentage(origin_stats.get('pct_ad_cvd_possible'))}

## Proof replay integrity summary
- Determinism check: **{determinism_status}**
- Proof payload hash stability: {_format_percentage(100.0 if determinism.get('passed') else 0.0)}

## Risk distribution summary
{chr(10).join(risk_lines) if risk_lines else 'No risk grades recorded.'}

## Known limitations
{chr(10).join(f"- {item}" for item in limitations) if limitations else '- None recorded'}

## Top 10 improvements
{chr(10).join(improvements)}

## Top 10 SKUs by annual savings
{chr(10).join(top_line_items) if top_line_items else 'No savings computed.'}

"""

    report_path.write_text(content, encoding="utf-8")
    print(f"Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
