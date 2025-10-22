import os

os.environ["CALE_EMBED_BACKEND"] = "hash"

from potatobacon.cale.embed import FeatureEngine, JURISDICTIONS
from potatobacon.cale.parser import PredicateMapper, RuleParser


def _parse(text: str, **meta):
    parser = RuleParser(PredicateMapper())
    metadata = {
        "id": meta.get("id", "R0"),
        "jurisdiction": meta.get("jurisdiction", "CA.Federal"),
        "statute": meta.get("statute", "X"),
        "section": meta.get("section", "1"),
        "enactment_year": meta.get("year", 2000),
    }
    return parser.parse(text, metadata)


def test_shapes_and_determinism():
    engine = FeatureEngine()
    rule = _parse("Organizations MUST obtain consent before collecting data", year=2001)
    enriched = engine.populate(rule)

    assert enriched.situational_vec.shape == (4,)
    assert enriched.interpretive_vec.shape == (384,)
    assert isinstance(enriched.temporal_scalar, float)
    assert enriched.jurisdictional_vec.shape == (len(JURISDICTIONS),)

    again = engine.populate(rule)
    assert (enriched.interpretive_vec == again.interpretive_vec).all()


def test_temporal_scalar_monotonic():
    engine = FeatureEngine()
    early = engine.populate(_parse("X MUST do Y", year=1950))
    late = engine.populate(_parse("X MUST do Y", year=2020))
    assert early.temporal_scalar < late.temporal_scalar


def test_jurisdiction_one_hot():
    engine = FeatureEngine()
    bc = engine.populate(_parse("X MUST do Y", jurisdiction="CA.BC"))
    us = engine.populate(_parse("X MUST do Y", jurisdiction="US.Federal"))
    assert abs(bc.jurisdictional_vec.sum() - 1.0) < 1e-6
    assert bc.jurisdictional_vec.argmax() != us.jurisdictional_vec.argmax()
