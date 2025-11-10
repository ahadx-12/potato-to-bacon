from __future__ import annotations

import datetime as dt

from potatobacon.law.ambiguity_entropy import normalized_entropy
from potatobacon.law.contradiction_score import ContradictionFeatures, contradiction_probability
from potatobacon.law.features_hierarchy import hierarchy_term, HierarchyContext
from potatobacon.law.flaw_score import flaw_score, policy_flaw_score
from potatobacon.law.impact_weight import impact_weight
from potatobacon.law.judicial_uncertainty import judicial_uncertainty
from potatobacon.law.network_fragility import compute_network_scores


def test_hierarchy_term_weights_recent_statute_vs_regulation() -> None:
    context = HierarchyContext(
        layer_a="statute",
        layer_b="regulation",
        shared_citations=["case1", "case2"],
        citation_total=4,
        newer_date=dt.date(2023, 5, 1),
        older_date=dt.date(2019, 5, 1),
    )
    value = hierarchy_term(context)
    assert 0.2 < value < 1.0


def test_contradiction_probability_increases_with_conflicts() -> None:
    features = ContradictionFeatures(
        cosine_similarity=0.9,
        negation_overlap=0.8,
        numeric_conflict=1.0,
        shared_citations=["usc:26:1"],
        citation_total=1,
        layer_a="statute",
        layer_b="regulation",
        newer_date=dt.date(2024, 1, 1),
        older_date=dt.date(2020, 1, 1),
    )
    probability = contradiction_probability(features)
    assert 0.5 < probability <= 1.0


def test_normalized_entropy_bounds() -> None:
    assert normalized_entropy([0.5, 0.5]) == 1.0
    assert normalized_entropy([1.0, 0.0]) == 0.0


def test_judicial_uncertainty_nonnegative() -> None:
    score = judicial_uncertainty([0.2, 0.4, 0.9], split_circuits=1, total_circuits=2)
    assert score >= 0.0


def test_network_fragility_scores_sum_positive() -> None:
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    scores = compute_network_scores(edges)
    assert set(scores.keys()) == {"A", "B", "C"}
    assert all(value >= 0.0 for value in scores.values())


def test_impact_and_flaw_scores() -> None:
    weight = impact_weight(1000, 0.8)
    assert weight > 0
    flaw = flaw_score(0.5, 0.6, 0.4, 0.2)
    assert flaw > 0
    policy = policy_flaw_score(flaw, weight)
    assert policy >= flaw
