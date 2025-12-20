from pathlib import Path

from potatobacon.tariff.analysis_session_store import AnalysisSessionStore
from potatobacon.tariff.sku_models import FactOverrideModel


def test_analysis_session_store_roundtrip(tmp_path: Path):
    store_path = tmp_path / "sessions.jsonl"
    store = AnalysisSessionStore(store_path)

    session = store.create_session("SKU-SESSION", law_context="CTX-DEMO")
    assert session.session_id
    assert session.sku_id == "SKU-SESSION"

    override = {"origin_country_US": FactOverrideModel(value=True, source="certificate", evidence_ids=["abc"])}
    updated = store.update_session(
        session.session_id,
        fact_overrides=override,
        attached_evidence_ids=["abc", "def"],
        status="READY_TO_OPTIMIZE",
    )

    assert updated.attached_evidence_ids == ["abc", "def"]
    assert "origin_country_US" in updated.fact_overrides
    assert updated.status == "READY_TO_OPTIMIZE"

    # Idempotent reapply should not bump timestamps or reorder serialized output.
    unchanged = store.update_session(
        session.session_id,
        fact_overrides=override,
        attached_evidence_ids=["def", "abc"],
        status="READY_TO_OPTIMIZE",
    )
    assert unchanged.updated_at == updated.updated_at

    reload = AnalysisSessionStore(store_path)
    restored = reload.get(session.session_id)
    assert restored is not None
    assert restored.fact_overrides["origin_country_US"].source == "certificate"
    assert restored.attached_evidence_ids == ["abc", "def"]

    lines = store_path.read_text().strip().splitlines()
    assert len(lines) == 1
