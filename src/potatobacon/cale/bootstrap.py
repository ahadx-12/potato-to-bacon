"""Unified bootstrap for CALE services.

This module constructs deterministic, in-memory service objects shared by both the
API and CLI entrypoints.  It purposely avoids performing heavyweight work at
module import time so that call-sites explicitly decide when initialisation
occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import json
import logging
import os

from .parser import PredicateMapper, RuleParser
from .symbolic import SymbolicConflictChecker
from .embed import LegalEmbedder, FeatureEngine
from .precedent import PrecedentIndex, load_precedent_cases
from .graph import load_citation_graph, compute_authority_scores
from .ccs import CCSCalculator
from .types import LegalRule

LOGGER = logging.getLogger(__name__)


def ensure_demo_corpus(path: Path) -> None:
    """Create a minimal demo corpus if the expected file is missing."""

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    demo = [
        {
            "id": "PIPEDA_7_3",
            "title": "Personal Information Protection and Electronic Documents Act, Section 7(3)",
            "jurisdiction": "Canada",
            "text": "An organization MUST obtain consent before collecting personal data.",
            "citations": ["ANTI_TERRORISM_83_28"],
        },
        {
            "id": "ANTI_TERRORISM_83_28",
            "title": "Anti-Terrorism Act, Section 83.28",
            "jurisdiction": "Canada",
            "text": "A government agency MAY collect personal data without consent in cases of national security.",
            "citations": ["PIPEDA_7_3"],
        },
    ]
    path.write_text(json.dumps(demo, indent=2), encoding="utf-8")
    print(f"[CALE Bootstrap] Created demo corpus → {path}")


@dataclass
class CALEServices:
    mapper: PredicateMapper
    parser: RuleParser
    checker: SymbolicConflictChecker
    embedder: LegalEmbedder
    feature_engine: FeatureEngine
    calculator: CCSCalculator
    suggester: object  # AmendmentSuggester
    authority: Dict[str, float]
    corpus: List[LegalRule]
    precedent_index: PrecedentIndex | None = None

    @property
    def symbolic(self) -> SymbolicConflictChecker:
        return self.checker

    @property
    def ccs(self) -> CCSCalculator:
        return self.calculator

    @property
    def authority_scores(self) -> Dict[str, float]:
        return self.authority


def _load_corpus(path: str, parser: RuleParser | None = None) -> List[LegalRule]:
    dataset_path = os.fspath(path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CALE corpus missing at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as handle:
        items: Sequence[dict] = json.load(handle)

    rules: List[LegalRule] = []
    for item in items:
        payload = dict(item)
        if {"subject", "modality", "action", "conditions"}.issubset(payload):
            try:
                rules.append(LegalRule(**payload))
                continue
            except TypeError as exc:
                LOGGER.debug("Failed to coerce raw payload into LegalRule: %s", exc)

        if parser is None:
            raise ValueError("Parser required to coerce corpus entry into LegalRule")
        metadata = {
            "id": payload.get("id"),
            "jurisdiction": payload.get("jurisdiction", ""),
            "statute": payload.get("statute", ""),
            "section": payload.get("section", ""),
            "enactment_year": payload.get("enactment_year", payload.get("year", 1900)),
        }
        try:
            rule = parser.parse(payload["text"], metadata)
        except Exception as exc:  # pragma: no cover - defensive logging for malformed rows
            LOGGER.warning("Skipping malformed CALE corpus entry %s: %s", payload.get("id"), exc)
            continue
        rules.append(rule)

    return rules


def build_services(corpus_path: str | None = None, deterministic: bool = True) -> CALEServices:
    corpus_path = corpus_path or os.getenv("CALE_CORPUS_PATH", "data/cale/demo_corpus.json")
    ensure_demo_corpus(Path(corpus_path))

    mapper = PredicateMapper()
    parser = RuleParser(predicate_mapper=mapper)
    checker = SymbolicConflictChecker(predicate_mapper=mapper)

    embedder = LegalEmbedder(model_name=None, deterministic=deterministic)

    graph = load_citation_graph(corpus_path)
    authority = compute_authority_scores(graph)

    feature_engine = FeatureEngine(embedder=embedder, authority_scores=authority)

    raw_corpus = _load_corpus(corpus_path, parser=parser)
    populated = [feature_engine.populate(rule) for rule in raw_corpus]

    calculator = CCSCalculator(prefer_torch=not os.getenv("CALE_NUMPY_ONLY"))

    from .suggest import AmendmentSuggester

    suggester = AmendmentSuggester(
        rule_corpus=populated,
        embedder=embedder,
        ccs_calculator=calculator,
        predicate_mapper=mapper,
    )

    precedent_index: PrecedentIndex | None = None
    precedent_path = os.getenv("CALE_PRECEDENT_PATH", "data/cale/precedents.json")
    try:
        if Path(precedent_path).exists():
            cases = load_precedent_cases(precedent_path, embedder=embedder)
            precedent_index = PrecedentIndex(cases)
    except Exception as exc:  # pragma: no cover - logging only
        LOGGER.warning("Failed to load precedent index from %s: %s", precedent_path, exc)

    return CALEServices(
        mapper=mapper,
        parser=parser,
        checker=checker,
        embedder=embedder,
        feature_engine=feature_engine,
        calculator=calculator,
        suggester=suggester,
        authority=authority,
        corpus=populated,
        precedent_index=precedent_index,
    )
