from potatobacon.tariff.atoms_hts import tariff_policy_atoms
from potatobacon.tariff.models import BaselineCandidateModel
from potatobacon.tariff.questions import generate_missing_fact_questions


def test_generate_missing_fact_questions_deterministic():
    atoms = tariff_policy_atoms()
    target_atom = next(atom for atom in atoms if atom.source_id == "HTS_6404_19_35")
    candidate = BaselineCandidateModel(
        candidate_id=target_atom.source_id,
        active_codes=[target_atom.source_id],
        duty_rate=3.0,
        provenance_chain=[],
        confidence=0.5,
        missing_facts=["surface_contact_textile_gt_50"],
        compliance_flags={},
    )

    compiled_facts = {"requires_origin_data": True}
    first = generate_missing_fact_questions(
        law_context="HTS_US_DEMO_2025",
        atoms=atoms,
        compiled_facts=compiled_facts,
        candidates=[candidate],
    )
    second = generate_missing_fact_questions(
        law_context="HTS_US_DEMO_2025",
        atoms=atoms,
        compiled_facts=compiled_facts,
        candidates=[candidate],
    )

    assert first.missing_facts == sorted(set(first.missing_facts))
    assert first.missing_facts == second.missing_facts
    assert "surface_contact_textile_gt_50" in first.missing_facts
    assert "origin_country" in first.missing_facts

    surface_question = next(item for item in first.questions if item.fact_key == "surface_contact_textile_gt_50")
    assert "50%" in surface_question.question
    assert surface_question.candidate_rules_affected
    assert first.model_dump() == second.model_dump()
