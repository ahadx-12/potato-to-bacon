from potatobacon.tariff.mutation_generator import generate_mutation_candidates
from tests.data.product_specs import bolt_spec, footwear_spec


def test_fastener_generates_aluminum_swap():
    candidates = generate_mutation_candidates(bolt_spec)
    descriptions = {candidate.human_description for candidate in candidates}
    assert any("aluminum" in desc.lower() for desc in descriptions)


def test_footwear_generates_felt_overlay():
    candidates = generate_mutation_candidates(footwear_spec)
    patches = [candidate.mutation_patch for candidate in candidates]
    assert any(patch.get("surface_coverage") for patch in patches)
