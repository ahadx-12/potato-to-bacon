from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from potatobacon.proofs.canonical import canonical_json
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID
from potatobacon.tariff.sku_dossier import build_sku_dossier_v2
from potatobacon.tariff.sku_store import SKUStore


SMOKE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "sku_id": "RW-ELEC-USB-CABLE-001",
        "description": "USB-C braided cable assembly with connector pair; copper conductors, no explicit insulation data",
        "origin_country": "VN",
        "declared_value_per_unit": 3.2,
        "annual_volume": 120000,
    },
    {
        "sku_id": "RW-ELEC-HDMI-CABLE-001",
        "description": "HDMI video cable harness with molded plastic plugs and braided jacket; low-voltage signal class",
        "origin_country": "CN",
        "declared_value_per_unit": 4.8,
        "annual_volume": 85000,
    },
]


def _report_path() -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("reports") / f"CALE-TARIFF_REALWORLD_SMOKE_{timestamp}.md"


def _persist_scenarios(store: SKUStore) -> None:
    for scenario in SMOKE_SCENARIOS:
        payload = {key: value for key, value in scenario.items() if key != "sku_id"}
        store.upsert(scenario["sku_id"], payload)


def _render_candidates(dossier) -> List[str]:
    lines: List[str] = []
    for candidate in dossier.baseline.candidates:
        missing = ", ".join(candidate.missing_facts) if candidate.missing_facts else "none"
        lines.append(f"- {candidate.candidate_id} @ {candidate.duty_rate}% (missing: {missing})")
    return lines


def _render_lever_outcome(dossier) -> str:
    if dossier.optimized and dossier.optimized.suggestion:
        suggestion = dossier.optimized.suggestion
        return (
            f"- Lever: {suggestion.lever_id} → {suggestion.optimized_duty_rate}% "
            f"(from {suggestion.baseline_duty_rate}%); savings/unit={suggestion.savings_per_unit_value:.4f}"
        )
    return "- Lever: none (baseline only)"


def main() -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    store = SKUStore(Path("data/realworld_smoke_skus.jsonl"))
    _persist_scenarios(store)

    report_lines: List[str] = []
    report_lines.append("# CALE-TARIFF Real-World Smoke")
    report_lines.append("")
    report_lines.append(f"- Law context: {DEFAULT_CONTEXT_ID}")
    report_lines.append(f"- Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    report_lines.append("")

    for scenario in SMOKE_SCENARIOS:
        dossier = build_sku_dossier_v2(
            scenario["sku_id"],
            law_context=DEFAULT_CONTEXT_ID,
            evidence_requested=False,
            optimize=True,
            store=store,
        )
        report_lines.append(f"## {scenario['sku_id']} — {dossier.status}")
        report_lines.append(f"- Description: {scenario['description']}")
        report_lines.append(f"- Baseline candidates ({len(dossier.baseline.candidates)}):")
        report_lines.extend(_render_candidates(dossier))
        missing = ", ".join(dossier.questions.missing_facts) if dossier.questions.missing_facts else "none"
        report_lines.append(f"- Missing facts: {missing}")
        report_lines.append(_render_lever_outcome(dossier))
        report_lines.append(f"- Why not optimized: {', '.join(dossier.why_not_optimized) or 'n/a'}")
        report_lines.append(f"- Tariff manifest: {dossier.tariff_manifest_hash}")
        report_lines.append(f"- Proof payload hash: {dossier.proof_payload_hash or 'n/a'}")
        report_lines.append("")
        print(
            canonical_json(
                {
                    "sku_id": scenario["sku_id"],
                    "status": dossier.status,
                    "baseline_candidates": [cand.model_dump() for cand in dossier.baseline.candidates],
                    "missing_facts": dossier.questions.missing_facts,
                    "lever": dossier.optimized.suggestion.lever_id
                    if dossier.optimized and dossier.optimized.suggestion
                    else None,
                }
            )
        )

    output_path = _report_path()
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Real-world smoke report written to {output_path}")


if __name__ == "__main__":
    main()
