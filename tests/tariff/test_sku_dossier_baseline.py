from potatobacon.tariff.sku_dossier import build_sku_dossier_v2
from potatobacon.tariff.sku_store import get_default_sku_store


def test_baseline_assignment_prefers_satisfied_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    sku_store = get_default_sku_store(tmp_path / "skus.jsonl")
    sku_id = "SKU-USB-BASELINE"
    sku_store.upsert(
        sku_id,
        {
            "sku_id": sku_id,
            "description": "USB-C cable assembly with dual connectors; low-voltage signal class",
            "declared_value_per_unit": 5.5,
            "annual_volume": 120000,
        },
    )

    fact_overrides = {
        "product_type_electronics": {"value": True, "source": "test"},
        "electronics_cable_or_connector": {"value": True, "source": "test"},
        "electronics_has_connectors": {"value": True, "source": "test"},
        "electronics_is_cable_assembly": {"value": True, "source": "test"},
        "electronics_voltage_rating_known": {"value": True, "source": "test"},
    }

    dossier = build_sku_dossier_v2(
        sku_id,
        optimize=False,
        evidence_requested=False,
        fact_overrides=fact_overrides,
        store=sku_store,
    )
    assert dossier.baseline_assigned is not None
    assert dossier.baseline_assigned.atom_id == "HTS_ELECTRONICS_CONNECTOR"

    connector_candidate = next(
        cand for cand in dossier.baseline.candidates if cand.candidate_id == dossier.baseline_assigned.atom_id
    )
    assert connector_candidate.missing_facts == []

    pathway = next(path for path in dossier.conditional_pathways if path.atom_id == "HTS_ELECTRONICS_SIGNAL_LOW_VOLT")
    assert "electronics_insulated_conductors" in pathway.missing_facts
    assert "spec_sheet" in pathway.accepted_evidence_types
