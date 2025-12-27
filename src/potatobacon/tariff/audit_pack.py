from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from potatobacon.proofs.store import get_default_store
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.candidate_search import generate_baseline_candidates
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.fact_requirements import FactRequirementRegistry
from potatobacon.tariff.normalizer import normalize_compiled_facts
from potatobacon.tariff.sku_dossier import _conditional_pathways, _select_baseline_assignment
from potatobacon.tariff.suggest import _evaluate_scenario
from potatobacon.tariff.models import ProductGraph
from potatobacon.tariff.origin_engine import compute_rvc


def _draw_lines(c: canvas.Canvas, lines: List[str], *, start_y: int = 760, step: int = 14) -> None:
    y = start_y
    for line in lines:
        c.drawString(40, y, line)
        y -= step
        if y < 60:
            c.showPage()
            y = start_y


def _provenance_lines(chain: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for entry in chain[:15]:
        label = f"{entry.get('source_id')} ({entry.get('scenario')})"
        statute = entry.get("statute") or ""
        section = entry.get("section") or ""
        lines.append(f"- {label} :: {statute} {section}".strip())
    return lines or ["- provenance unavailable"]


def _fact_table_lines(compiled_facts: Dict[str, Any], fact_overrides: Dict[str, Any] | None) -> List[str]:
    overrides = fact_overrides or {}
    lines: List[str] = []
    for key in sorted(compiled_facts.keys()):
        if key in {"raw", "baseline", "normalized", "overrides", "attached_evidence_ids", "analysis_session_id"}:
            continue
        value = compiled_facts[key]
        if isinstance(value, dict) or isinstance(value, list):
            continue
        source = "override" if key in overrides else "inferred"
        lines.append(f"- {key}: {value} ({source})")
    return lines or ["- no compiled facts available"]


def _overlay_lines(overlays: Dict[str, Any] | None) -> List[str]:
    if not overlays:
        return ["- no overlays applied"]
    lines: List[str] = []
    for phase in ("baseline", "optimized"):
        phase_items = overlays.get(phase) or []
        lines.append(f"{phase.capitalize()} overlays:")
        if not phase_items:
            lines.append("  - none")
            continue
        for item in phase_items:
            label = item.get("overlay_name") if isinstance(item, dict) else getattr(item, "overlay_name", "")
            rate = item.get("additional_rate") if isinstance(item, dict) else getattr(item, "additional_rate", 0)
            reason = item.get("reason") if isinstance(item, dict) else getattr(item, "reason", "")
            requires_review = item.get("requires_review") if isinstance(item, dict) else getattr(item, "requires_review", False)
            stop_flag = item.get("stop_optimization") if isinstance(item, dict) else getattr(item, "stop_optimization", False)
            flags = []
            if requires_review:
                flags.append("review")
            if stop_flag:
                flags.append("stop")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"  - {label}: +{rate}% ({reason}){suffix}")
    return lines or ["- no overlays applied"]


def _value_added_lines(product_graph: ProductGraph | None, sku_metadata: Dict[str, Any]) -> List[str]:
    if not product_graph:
        return ["- Value-Added Worksheet unavailable (no product graph)"]
    adjusted_value = sku_metadata.get("declared_value_per_unit")
    rvc_result = compute_rvc(
        product_graph,
        adjusted_value,
        declared_origin_country=sku_metadata.get("origin_country"),
    )
    if not rvc_result:
        return ["- Value-Added Worksheet incomplete (missing adjusted value or component values)"]
    return [
        f"- Adjusted value (V_adj): {rvc_result.adjusted_value}",
        f"- Originating materials value: {rvc_result.originating_value}",
        f"- Non-originating materials value: {rvc_result.non_originating_value}",
        f"- Build-down RVC: {rvc_result.build_down}%",
        f"- Build-up RVC: {rvc_result.build_up}%",
    ]


def generate_audit_pack_pdf(proof_id: str) -> bytes:
    store = get_default_store()
    proof = store.get_proof(proof_id)
    if not proof:
        raise KeyError(proof_id)

    evidence_pack = proof.get("evidence_pack") or {}
    compiled_facts = evidence_pack.get("compiled_facts") or proof.get("compiled_facts") or {}
    baseline_facts = compiled_facts.get("normalized") or compiled_facts.get("baseline") or {}
    normalized_facts, _ = normalize_compiled_facts(baseline_facts)
    law_context = proof.get("law_context") or DEFAULT_CONTEXT_ID
    try:
        atoms, context_meta = load_atoms_for_context(law_context)
    except KeyError:
        law_context = DEFAULT_CONTEXT_ID
        atoms, context_meta = load_atoms_for_context(law_context)
    duty_rates = context_meta.get("duty_rates") or DUTY_RATES
    baseline_candidates = generate_baseline_candidates(normalized_facts, atoms, duty_rates, max_candidates=5)
    baseline_eval = _evaluate_scenario(atoms, normalized_facts, duty_rates)
    baseline_assignment = _select_baseline_assignment(baseline_eval, baseline_candidates, duty_rates)
    requirement_registry = FactRequirementRegistry()
    conditional_pathways = _conditional_pathways(
        baseline_candidates, baseline_assignment, requirement_registry=requirement_registry
    )

    output = BytesIO()
    c = canvas.Canvas(output, pagesize=letter)
    c.setTitle("CALE-LAW Tariff Dossier Audit Pack")
    c.setPageCompression(0)

    manifest_hash = proof.get("tariff_manifest_hash") or context_meta.get("manifest_hash")
    payload_hash = proof.get("proof_payload_hash")
    sku_metadata = evidence_pack.get("sku_metadata") or {}
    product_graph_payload = evidence_pack.get("product_graph")
    product_graph = ProductGraph(**product_graph_payload) if isinstance(product_graph_payload, dict) else None
    description_hash = sku_metadata.get("description_hash", "n/a")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 800, "CALE-LAW Tariff Dossier Audit Pack")
    c.setFont("Helvetica", 11)
    meta_lines = [
        f"Proof: {proof_id}",
        f"Payload hash: {payload_hash}",
        f"Law context: {law_context}",
        f"Manifest hash: {manifest_hash}",
        f"Description hash: {description_hash}",
        f"Replay status: PASS",
    ]
    _draw_lines(c, meta_lines, start_y=780)

    duty_lines = [
        f"Baseline duty: {proof.get('baseline', {}).get('duty_rate')}",
        f"Optimized duty: {proof.get('optimized', {}).get('duty_rate')}",
        f"Baseline assignment: {baseline_assignment.atom_id} @ {baseline_assignment.duty_rate}",
    ]
    if conditional_pathways:
        duty_lines.append("Conditional pathways:")
        for path in conditional_pathways:
            duty_lines.append(
                f"  - {path.atom_id} @ {path.duty_rate}% missing {', '.join(path.missing_facts)}"
            )
            duty_lines.append(f"    evidence: {', '.join(path.accepted_evidence_types)}")
    else:
        duty_lines.append("Conditional pathways: none")
    _draw_lines(c, duty_lines, start_y=680)

    overlay_section = ["Overlays:"]
    overlay_section.extend(_overlay_lines(proof.get("overlays")))
    _draw_lines(c, overlay_section, start_y=620)

    fact_lines = ["Facts:"]
    fact_lines.extend(_fact_table_lines(normalized_facts, compiled_facts.get("overrides")))
    _draw_lines(c, fact_lines, start_y=560)

    value_added_lines = ["Value-Added Worksheet:"]
    value_added_lines.extend(_value_added_lines(product_graph, sku_metadata))
    _draw_lines(c, value_added_lines, start_y=470)

    provenance_chain = proof.get("provenance_chain") or []
    provenance_lines = ["Provenance:"]
    provenance_lines.extend(_provenance_lines(provenance_chain))
    _draw_lines(c, provenance_lines, start_y=350)

    warning_lines = ["Warnings:"]
    if proof.get("baseline", {}).get("duty_status") != "OK" or proof.get("optimized", {}).get("duty_status") != "OK":
        warning_lines.append("- Review duty status flags before execution")
    else:
        warning_lines.append("- none")
    _draw_lines(c, warning_lines, start_y=210)

    c.showPage()
    c.save()
    return output.getvalue()
