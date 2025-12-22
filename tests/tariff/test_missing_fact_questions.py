from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.models import BaselineCandidateModel
from potatobacon.tariff.questions import generate_missing_fact_questions
from potatobacon.tariff.sku_dossier import build_sku_dossier_v2
from potatobacon.tariff.sku_store import get_default_sku_store


def test_missing_fact_explains_atoms_and_levers(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    atoms = [
        PolicyAtom(
            guard=["electronics_voltage_rating_known"],
            outcome={"action": "duty_rate"},
            source_id="HTS_ELECTRONICS_SIGNAL_LOW_VOLT",
            text="Electronics cable under low voltage class",
        )
    ]
    candidates = [
        BaselineCandidateModel(
            candidate_id="HTS_ELECTRONICS_SIGNAL_LOW_VOLT",
            active_codes=["HTS_ELECTRONICS_SIGNAL_LOW_VOLT"],
            duty_rate=3.0,
            provenance_chain=[],
            confidence=0.9,
            missing_facts=["electronics_voltage_rating_known"],
        )
    ]
    lever_requirements = {"electronics_voltage_rating_known": ["ELECTRONICS_CABLE_ASSEMBLY_PATHWAY"]}

    questions = generate_missing_fact_questions(
        law_context="TEST",
        atoms=atoms,
        compiled_facts={},
        candidates=candidates,
        lever_requirements=lever_requirements,
    )
    question = questions.questions[0]
    assert "HTS_ELECTRONICS_SIGNAL_LOW_VOLT" in question.why_needed
    assert "ELECTRONICS_CABLE_ASSEMBLY_PATHWAY" in question.why_needed
    assert question.blocks_classification is True
    assert question.blocks_optimization is True


def test_dossier_missing_fact_templates_present(monkeypatch, tmp_path):
    monkeypatch.setenv("PTB_DATA_ROOT", str(tmp_path))
    sku_store = get_default_sku_store()
    sku_id = "SKU-MISSING-TEMPLATES"
    sku_store.upsert(
        sku_id,
        {
            "sku_id": sku_id,
            "description": "Electronics enclosure with harness and PCB slot",
            "declared_value_per_unit": 55.0,
        },
    )

    dossier = build_sku_dossier_v2(sku_id, optimize=False, evidence_requested=False)
    assert dossier.questions.missing_facts
    for question in dossier.questions.questions:
        assert question.accepted_evidence_types, f"missing evidence template for {question.fact_key}"
        assert question.why_needed
