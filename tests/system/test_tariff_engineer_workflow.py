import pytest

from potatobacon.tariff.evidence_store import get_default_evidence_store
from potatobacon.tariff.sku_dossier import build_sku_dossier_v2
from potatobacon.tariff.sku_store import get_default_sku_store


@pytest.mark.usefixtures("system_app")
def test_tariff_engineer_workflow_audit_loop():
    sku_id = "SKU-ENGINEER-AUDIT-1"
    sku_store = get_default_sku_store()
    sku_store.upsert(
        sku_id,
        {
            "sku_id": sku_id,
            "description": "Signal cable assembly with dual connectors and braided jacket",
            "declared_value_per_unit": 9.5,
            "annual_volume": 12000,
            "origin_country": "VN",
        },
    )

    baseline = build_sku_dossier_v2(sku_id, optimize=True, evidence_requested=True)
    assert any(
        entry.status.value == "BLOCKED_MISSING_FACTS"
        for entry in baseline.opportunity_ledger.entries
    )
    bundle_labels = {item.request_label for item in baseline.intake_bundle.items}
    assert "BOM CSV" in bundle_labels
    assert "Technical Spec" in bundle_labels

    bom_csv = """part_name,material,origin_country,value
usb cable jacket,plastic,VN,1.5
signal core,copper,VN,2.0
"""
    evidence_store = get_default_evidence_store()
    evidence_record = evidence_store.save(
        bom_csv.encode("utf-8"),
        filename="usb_bom.csv",
        content_type="text/csv",
        evidence_kind="bom_csv",
    )

    refined = build_sku_dossier_v2(
        sku_id,
        optimize=True,
        evidence_requested=True,
        attached_evidence_ids=[evidence_record.evidence_id],
    )
    assert any(
        ev.fact_key == "electronics_insulated_conductors"
        and any(tag.startswith("evidence:") for tag in ev.derived_from)
        for ev in refined.fact_evidence or []
    )

    harness_lane = next(
        entry
        for entry in refined.opportunity_ledger.entries
        if entry.lane_id == "HTS_ELECTRONICS_WIRE_HARNESS"
    )
    assert harness_lane.status.value == "AVAILABLE_NOW"

    replay = build_sku_dossier_v2(
        sku_id,
        optimize=True,
        evidence_requested=True,
        attached_evidence_ids=[evidence_record.evidence_id],
    )
    assert refined.proof_payload_hash
    assert refined.proof_payload_hash == replay.proof_payload_hash

    assert refined.optimized
    suggestion = refined.optimized.suggestion
    assert suggestion
    assert suggestion.precedent_context
    assert suggestion.precedent_context["matched_rulings"]
