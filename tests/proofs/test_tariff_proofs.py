import importlib

from potatobacon.tariff.models import TariffHuntRequestModel


BASELINE_FACTS = {
    "upper_material_textile": True,
    "outer_sole_material_rubber_or_plastics": True,
    "surface_contact_rubber_gt_50": True,
    "surface_contact_textile_gt_50": False,
    "felt_covering_gt_50": False,
}


def test_tariff_proof_persisted(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))

    import potatobacon.proofs.store as store_mod
    import potatobacon.proofs.engine as proofs_engine
    import potatobacon.tariff.engine as engine_mod

    store_mod = importlib.reload(store_mod)
    proofs_engine = importlib.reload(proofs_engine)
    engine_mod = importlib.reload(engine_mod)

    request = TariffHuntRequestModel(
        scenario=BASELINE_FACTS,
        mutations={"felt_covering_gt_50": True},
        seed=2025,
    )
    dossier = engine_mod.run_tariff_hack(
        base_facts=request.scenario,
        mutations=request.mutations,
        law_context=request.law_context,
        seed=request.seed or 2025,
    )

    store = store_mod.get_default_store()
    proof_record = store.get_proof(dossier.proof_id)
    assert proof_record is not None
    assert proof_record["proof_id"] == dossier.proof_id
    assert proof_record["law_context"] == dossier.law_context
    assert proof_record["input"]["scenario"]["felt_covering_gt_50"] is False
