"""Runtime bootstrap for CALE services.

This module initialises the deterministic CALE service graph exactly once per
process.  The bootstrap routine is idempotent and caches the constructed
services so that API routes, CLI commands, and tests can share a single set of
singletons without dealing with implicit globals scattered across modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .parser import PredicateMapper, RuleParser
from .symbolic import SymbolicConflictChecker
from .embed import FeatureEngine, LegalEmbedder
from .graph import compute_authority_scores, load_citation_graph
from .ccs import CCSCalculator
from .types import LegalRule
from .suggest import AmendmentSuggester


@dataclass
class CALEServices:
    """Container bundling all runtime singletons for CALE."""

    predicate_mapper: PredicateMapper
    parser: RuleParser
    symbolic: SymbolicConflictChecker
    embedder: LegalEmbedder
    feature_engine: FeatureEngine
    ccs: CCSCalculator
    corpus: List[LegalRule]
    authority_scores: Dict[str, float]
    suggester: AmendmentSuggester


_registry: Optional[CALEServices] = None


def _load_demo_corpus(path: str, parser: RuleParser) -> List[LegalRule]:
    """Load the demo corpus and parse each entry into a :class:`LegalRule`."""

    dataset_path = os.fspath(path)
    if not os.path.exists(dataset_path):
        return []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    rules: List[LegalRule] = []
    for record in raw:
        text = record.get("text")
        if not text:
            # Skip malformed entries to keep bootstrap resilient in tests.
            continue
        metadata = {
            "id": record.get("id"),
            "jurisdiction": record.get("jurisdiction", ""),
            "statute": record.get("statute", ""),
            "section": record.get("section", ""),
            "enactment_year": record.get("enactment_year", record.get("year", 1900)),
        }
        rule = parser.parse(text, metadata)
        rules.append(rule)
    return rules


def bootstrap(
    corpus_path: str = "data/cale/demo_corpus.json",
    disable: bool = False,
) -> Optional[CALEServices]:
    """Build all CALE singletons and cache them in memory.

    The function respects the ``CALE_DISABLE_STARTUP_INIT`` environment variable
    so that unit tests can opt out of the heavyweight initialisation path.  On
    success a :class:`CALEServices` instance is cached globally and returned; on
    failure the registry is cleared so callers can surface the error gracefully.
    """

    global _registry

    if disable or os.getenv("CALE_DISABLE_STARTUP_INIT") == "1":
        _registry = None
        return None

    if _registry is not None:
        return _registry

    predicate_mapper = PredicateMapper()
    parser = RuleParser(predicate_mapper)
    symbolic = SymbolicConflictChecker(predicate_mapper)
    embedder = LegalEmbedder()
    feature_engine = FeatureEngine(embedder)

    corpus = _load_demo_corpus(corpus_path, parser)
    graph = load_citation_graph(corpus_path)
    authority_scores = compute_authority_scores(graph)

    enriched_corpus: List[LegalRule] = []
    for rule in corpus:
        populated = feature_engine.populate_features(rule, authorities=authority_scores)
        enriched_corpus.append(populated)

    ccs = CCSCalculator()
    suggester = AmendmentSuggester(enriched_corpus, embedder, ccs, predicate_mapper)

    _registry = CALEServices(
        predicate_mapper=predicate_mapper,
        parser=parser,
        symbolic=symbolic,
        embedder=embedder,
        feature_engine=feature_engine,
        ccs=ccs,
        corpus=enriched_corpus,
        authority_scores=authority_scores,
        suggester=suggester,
    )
    return _registry


def get_services() -> CALEServices:
    """Return the cached services or raise if bootstrap has not run."""

    if _registry is None:
        raise RuntimeError("CALE runtime not initialised")
    return _registry


__all__ = ["CALEServices", "bootstrap", "get_services"]
