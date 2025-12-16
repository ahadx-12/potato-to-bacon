"""Search strategies for tariff redesign exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from .product_schema import ProductSpecModel


@dataclass
class SearchResult:
    mutated_product: ProductSpecModel
    mutations_applied: List[Dict[str, object]]
    annual_savings_value: float
    risk_score: float
    defensibility_grade: str


EvaluationFn = Callable[[ProductSpecModel], Tuple[float, float, str]]


def _apply_patch(product: ProductSpecModel, patch: Dict[str, object]) -> ProductSpecModel:
    # shallow update only; deterministic ordering preserved by ProductSpecModel
    updated_data = product.model_dump()
    for key, value in patch.items():
        updated_data[key] = value
    return ProductSpecModel(**updated_data)


def _score_candidate(
    mutated: ProductSpecModel, evaluate_fn: EvaluationFn
) -> SearchResult:
    savings, risk_score, defensibility = evaluate_fn(mutated)
    return SearchResult(
        mutated_product=mutated,
        mutations_applied=[],
        annual_savings_value=savings,
        risk_score=risk_score,
        defensibility_grade=defensibility,
    )


def cartesian_search(
    product: ProductSpecModel,
    mutation_groups: Sequence[Sequence[Dict[str, object]]],
    evaluate_fn: EvaluationFn,
    top_k: int = 5,
) -> List[SearchResult]:
    """Exhaustively search combinations of mutation patches."""

    results: List[SearchResult] = []

    def _recurse(idx: int, current_product: ProductSpecModel, applied: List[Dict[str, object]]):
        if idx == len(mutation_groups):
            scored = _score_candidate(current_product, evaluate_fn)
            scored.mutations_applied = list(applied)
            results.append(scored)
            return
        for patch in mutation_groups[idx]:
            mutated_product = _apply_patch(current_product, patch)
            _recurse(idx + 1, mutated_product, applied + [patch])

    _recurse(0, product, [])
    results.sort(key=lambda r: r.annual_savings_value, reverse=True)
    return results[:top_k]


def beam_search(
    product: ProductSpecModel,
    mutation_groups: Sequence[Sequence[Dict[str, object]]],
    evaluate_fn: EvaluationFn,
    beam_width: int = 3,
    top_k: int = 5,
) -> List[SearchResult]:
    """Beam search over mutation groups for medium search spaces."""

    frontier: List[Tuple[ProductSpecModel, List[Dict[str, object]]]] = [(product, [])]
    for patches in mutation_groups:
        candidates: List[Tuple[ProductSpecModel, List[Dict[str, object]], float]] = []
        for product_state, applied in frontier:
            for patch in patches:
                mutated = _apply_patch(product_state, patch)
                savings, risk_score, defensibility = evaluate_fn(mutated)
                candidates.append((mutated, applied + [patch], savings))
        candidates.sort(key=lambda item: item[2], reverse=True)
        frontier = [(prod, applied) for prod, applied, _ in candidates[:beam_width]]

    scored: List[SearchResult] = []
    for product_state, applied in frontier:
        savings, risk_score, defensibility = evaluate_fn(product_state)
        scored.append(
            SearchResult(
                mutated_product=product_state,
                mutations_applied=applied,
                annual_savings_value=savings,
                risk_score=risk_score,
                defensibility_grade=defensibility,
            )
        )
    scored.sort(key=lambda r: r.annual_savings_value, reverse=True)
    return scored[:top_k]


def arbitrage_hunter(
    product: ProductSpecModel,
    mutation_groups: Sequence[Sequence[Dict[str, object]]],
    evaluate_fn: EvaluationFn,
    iterations: int = 10,
    top_k: int = 5,
    beam_width: int = 3,
) -> List[SearchResult]:
    """Simplified genetic-style search that mutates top performers."""

    population = cartesian_search(product, mutation_groups[:1], evaluate_fn, top_k=beam_width)
    for _ in range(iterations):
        population = beam_search(
            product,
            mutation_groups,
            evaluate_fn,
            beam_width=min(len(mutation_groups), beam_width),
            top_k=top_k,
        )
    return population
