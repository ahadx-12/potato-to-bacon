from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.levers import (
    applicable_levers,
    diff_candidate_requirements,
    generate_candidate_levers,
)
from potatobacon.tariff.mutation_generator import baseline_facts_from_profile
from potatobacon.tariff.normalizer import normalize_compiled_facts
from potatobacon.tariff.product_schema import ProductCategory, ProductSpecModel
from potatobacon.tariff.suggest import _evaluate_scenario


def test_electronics_levers_require_material_and_enclosure():
    spec = ProductSpecModel(product_category=ProductCategory.ELECTRONICS)
    facts = {
        "product_type_electronics": True,
        "electronics_enclosure": True,
        "material_steel": True,
        "electronics_cable_or_connector": True,
        "electronics_has_connectors": True,
    }

    levers = applicable_levers(spec=spec, facts=facts)
    lever_ids = {lever.lever_id for lever in levers}
    assert "ELEC_ENCLOSURE_PLASTIC_DOMINANCE" in lever_ids
    assert "ELEC_CONNECTOR_PATHWAY" in lever_ids
    assert "ELECTRONICS_CABLE_ASSEMBLY_PATHWAY" in lever_ids


def test_apparel_lever_triggers_for_blend_near_threshold():
    spec = ProductSpecModel(product_category=ProductCategory.APPAREL_TEXTILE)
    facts = {
        "product_type_apparel_textile": True,
        "fiber_cotton_dominant": True,
        "fiber_polyester_dominant": False,
        "textile_woven": True,
    }

    levers = applicable_levers(spec=spec, facts=facts)
    lever_ids = {lever.lever_id for lever in levers}
    assert "APPAREL_BLEND_DOMINANCE" in lever_ids
    assert "APPAREL_CONFIRM_KNIT_WOVEN" not in lever_ids


def test_apparel_knit_confirmation_only_when_missing_flags():
    spec = ProductSpecModel(product_category=ProductCategory.APPAREL_TEXTILE)
    facts = {
        "product_type_apparel_textile": True,
    }
    levers = applicable_levers(spec=spec, facts=facts)
    lever_ids = {lever.lever_id for lever in levers}
    assert "APPAREL_CONFIRM_KNIT_WOVEN" in lever_ids


def _baseline_eval(facts: dict[str, object]):
    atoms, context_meta = load_atoms_for_context(DEFAULT_CONTEXT_ID)
    duty_rates = context_meta.get("duty_rates") or DUTY_RATES
    normalized_facts, _ = normalize_compiled_facts(facts)
    baseline_eval = _evaluate_scenario(atoms, normalized_facts, duty_rates)
    return atoms, duty_rates, normalized_facts, baseline_eval


def test_diff_candidate_requirements_detects_material_shift():
    atoms, _, _, _ = _baseline_eval({})
    baseline_atom = next(atom for atom in atoms if atom.source_id == "HTS_STEEL_BOLT")
    target_atom = next(atom for atom in atoms if atom.source_id == "HTS_ALUMINUM_PART")

    diffs = diff_candidate_requirements(baseline_atom, target_atom)

    assert diffs == ["material_aluminum"]


def test_candidate_lever_generated_for_fastener_material_change():
    facts = {
        "product_type_chassis_bolt": True,
        "material_steel": True,
        "material_aluminum": False,
    }
    atoms, duty_rates, normalized_facts, baseline_eval = _baseline_eval(facts)

    levers = generate_candidate_levers(
        baseline_atom=baseline_eval.duty_atoms[0],
        atoms=atoms,
        duty_rates=duty_rates,
        facts=normalized_facts,
        baseline_rate=baseline_eval.effective_duty_rate,
    )

    material_lever = next(lever for lever in levers if "DYNAMIC_MATERIAL" in lever.lever_id)
    assert material_lever.mutation["material_aluminum"] is True
    assert material_lever.mutation["material_steel"] is False


def test_candidate_lever_generated_for_surface_contact_shift():
    facts = baseline_facts_from_profile({"category": "footwear"})
    atoms, duty_rates, normalized_facts, baseline_eval = _baseline_eval(facts)

    levers = generate_candidate_levers(
        baseline_atom=baseline_eval.duty_atoms[0],
        atoms=atoms,
        duty_rates=duty_rates,
        facts=normalized_facts,
        baseline_rate=baseline_eval.effective_duty_rate,
    )

    overlay_lever = next(lever for lever in levers if "surface_contact_textile_gt_50" in lever.mutation)
    assert overlay_lever.mutation["surface_contact_textile_gt_50"] is True
    assert overlay_lever.mutation["surface_contact_rubber_gt_50"] is False
    assert set(overlay_lever.evidence_requirements) >= {"coverage_photo", "measurement_diagram"}


def test_dynamic_lever_generation_is_deterministic():
    facts = {
        "product_type_chassis_bolt": True,
        "material_steel": True,
        "material_aluminum": False,
    }
    atoms, duty_rates, normalized_facts, baseline_eval = _baseline_eval(facts)

    first = generate_candidate_levers(
        baseline_atom=baseline_eval.duty_atoms[0],
        atoms=atoms,
        duty_rates=duty_rates,
        facts=normalized_facts,
        baseline_rate=baseline_eval.effective_duty_rate,
    )
    second = generate_candidate_levers(
        baseline_atom=baseline_eval.duty_atoms[0],
        atoms=atoms,
        duty_rates=duty_rates,
        facts=normalized_facts,
        baseline_rate=baseline_eval.effective_duty_rate,
    )

    def _fingerprint(levers):
        return [
            (
                lever.lever_id,
                dict(lever.mutation),
                list(lever.evidence_requirements),
                lever.lever_type,
            )
            for lever in levers
        ]

    assert _fingerprint(first) == _fingerprint(second)


def test_missing_fact_diff_produces_documentation_lever():
    facts = {
        "product_type_chassis_bolt": True,
        "material_steel": True,
    }
    atoms, duty_rates, normalized_facts, baseline_eval = _baseline_eval(facts)

    levers = generate_candidate_levers(
        baseline_atom=baseline_eval.duty_atoms[0],
        atoms=atoms,
        duty_rates=duty_rates,
        facts=normalized_facts,
        baseline_rate=baseline_eval.effective_duty_rate,
    )

    doc_levers = [lever for lever in levers if lever.lever_type == "DOCUMENTATION"]
    assert doc_levers, "expected documentation lever for missing diff facts"
    assert any("material_aluminum" in lever.fact_gaps for lever in doc_levers)
    assert all("This does not change the product" in lever.rationale for lever in doc_levers)


def test_disallowed_dimensions_do_not_emit_levers():
    baseline_atom = PolicyAtom(
        guard=["origin_country"],
        outcome={"modality": "PERMIT", "action": "ALLOW"},
        source_id="ORIGIN_BASE",
    )
    candidate_atom = PolicyAtom(
        guard=["origin_country_export"],
        outcome={"modality": "PERMIT", "action": "ALLOW"},
        source_id="ORIGIN_TARGET",
    )

    levers = generate_candidate_levers(
        baseline_atom=baseline_atom,
        atoms=[baseline_atom, candidate_atom],
        duty_rates={"ORIGIN_BASE": 12.5, "ORIGIN_TARGET": 5.0},
        facts={},
        baseline_rate=12.5,
    )

    assert levers == []
