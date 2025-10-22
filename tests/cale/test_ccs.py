import os

os.environ["CALE_EMBED_BACKEND"] = "hash"

from potatobacon.cale.ccs import CCSCalculator
from potatobacon.cale.embed import FeatureEngine
from potatobacon.cale.parser import PredicateMapper, RuleParser


def _prep(text: str, rid: str, year: int):
    parser = RuleParser(PredicateMapper())
    metadata = {
        "id": rid,
        "jurisdiction": "CA.Federal",
        "statute": "X",
        "section": "1",
        "enactment_date": year,
    }
    rule = parser.parse(text, metadata)
    return FeatureEngine().populate(rule, authorities={rid: 0.5})


def test_ccs_range_and_ci_effect():
    rule1 = _prep("Org MUST collect data IF consent", "R1", 2000)
    rule2 = _prep("Org CANNOT collect data IF consent", "R2", 2000)
    calc = CCSCalculator()

    low = calc.compute_ccs(rule1, rule2, CI=0.0, philosophy="pragmatic")
    high = calc.compute_ccs(rule1, rule2, CI=1.0, philosophy="pragmatic")

    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high > low


def test_kernel_identity_and_philosophy_variance():
    rule1 = _prep("X MUST retain records", "R1", 1990)
    rule2 = _prep("X MUST retain records", "R2", 2020)
    calc = CCSCalculator()

    k_same = calc.compute_kernel(rule2.feature_vector, rule2.feature_vector, "pragmatic")
    assert 0.99 <= k_same <= 1.0

    ci = 0.0
    textualist = calc.compute_ccs(rule1, rule2, CI=ci, philosophy="textualist")
    living = calc.compute_ccs(rule1, rule2, CI=ci, philosophy="living")
    assert textualist > living
