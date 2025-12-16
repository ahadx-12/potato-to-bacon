from potatobacon.tariff.product_schema import ProductSpecModel
from potatobacon.tariff.search import beam_search, cartesian_search, arbitrage_hunter
from tests.data.product_specs import bolt_spec


def dummy_evaluate(product: ProductSpecModel):
    # Favor aluminum material and higher annual_volume
    material_names = {m.material.lower() for m in product.materials}
    savings = 0.0
    if "aluminum" in material_names:
        savings += 1.5
    savings += (product.annual_volume or 0) * 0.000001
    risk_score = 25.0 if "aluminum" in material_names else 50.0
    defensibility = "A" if risk_score <= 30 else "B"
    return savings, risk_score, defensibility


def test_cartesian_search_returns_sorted_results():
    results = cartesian_search(
        bolt_spec,
        mutation_groups=[[{"materials": [{"component": "body", "material": "aluminum"}]}, {}]],
        evaluate_fn=dummy_evaluate,
        top_k=2,
    )
    assert results
    assert results[0].annual_savings_value >= results[-1].annual_savings_value


def test_beam_search_limits_frontier():
    results = beam_search(
        bolt_spec,
        mutation_groups=[[{"materials": [{"component": "body", "material": "aluminum"}]}]],
        evaluate_fn=dummy_evaluate,
        beam_width=1,
    )
    assert results[0].mutations_applied


def test_arbitrage_hunter_runs_iterations():
    results = arbitrage_hunter(
        bolt_spec,
        mutation_groups=[[{"materials": [{"component": "body", "material": "aluminum"}]}]],
        evaluate_fn=dummy_evaluate,
        iterations=2,
        top_k=1,
        beam_width=1,
    )
    assert results
    assert results[0].mutations_applied
