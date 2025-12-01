from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
from fastapi.testclient import TestClient

os.environ["CALE_EMBED_BACKEND"] = "hash"
os.environ["CALE_API_KEYS"] = "test-key"

from potatobacon.api.app import app
from potatobacon.cale.ccs import CCSCalculator
from potatobacon.cale.embed import FeatureEngine, LegalEmbedder
from potatobacon.cale.parser import PredicateMapper, RuleParser
from potatobacon.cale.suggest import AmendmentSuggester
from potatobacon.cale.symbolic import SymbolicConflictChecker
from potatobacon.cale.types import LegalRule

_DEMO_RULE_TEXTS: Dict[str, str] = {
    "R1": "Organizations MUST collect personal data IF consent.",
    "R2": "Security agencies MUST NOT collect personal data IF emergency.",
    "R3": "Organizations MAY collect personal data IF emergency.",
    "R4": "Financial institutions MUST report personal data breaches IF data breach.",
    "R5": "Organizations MAY collect personal data IF national security threat.",
    "R6": "Provincial agencies MAY collect personal data IF court order.",
    "R7": "International agreements MAY require data disclosure IF treaty obligation.",
}


def _build_suggester() -> Tuple[AmendmentSuggester, Dict[str, LegalRule], CCSCalculator, SymbolicConflictChecker]:
    mapper = PredicateMapper()
    parser = RuleParser(mapper)
    embedder = LegalEmbedder()
    features = FeatureEngine(embedder)
    authorities = {rid: 0.5 for rid in _DEMO_RULE_TEXTS}
    corpus: Dict[str, LegalRule] = {}
    for idx, (rid, text) in enumerate(_DEMO_RULE_TEXTS.items(), start=1):
        metadata = {
            "id": rid,
            "jurisdiction": "CA.Federal",
            "statute": f"Statute-{idx}",
            "section": str(idx),
            "enactment_year": 1990 + idx,
        }
        rule = parser.parse(text, metadata)
        corpus[rid] = features.populate_features(rule, authorities=authorities)
    calc = CCSCalculator()
    suggester = AmendmentSuggester(list(corpus.values()), embedder, calc, mapper)
    symbolic = SymbolicConflictChecker(mapper)
    return suggester, corpus, calc, symbolic


def test_gradient_nonzero_and_shape() -> None:
    suggester, corpus, calc, symbolic = _build_suggester()
    r1 = corpus["R1"]
    r2 = corpus["R2"]
    ci = symbolic.check_conflict(r1, r2)
    grad = suggester._ccs_grad_x1(r1, r2, ci)
    assert grad.shape == r1.feature_vector.shape
    assert np.linalg.norm(grad) > 0


def test_knn_retrieval_filters() -> None:
    suggester, corpus, calc, symbolic = _build_suggester()
    r1 = corpus["R1"]
    r2 = corpus["R2"]
    ci = symbolic.check_conflict(r1, r2)
    analysis = calc.compute_multiperspective(r1, r2, ci)
    target = suggester._low_conflict_target(r1, r2, ci)
    precedents = suggester._find_precedents(target, r1, r2, analysis.CCS_pragmatic, ci, topk=3)
    assert len(precedents) <= 3
    for candidate in precedents:
        i1 = candidate.interpretive_vec
        i2 = r2.interpretive_vec
        cosine = float(np.dot(i1, i2) / ((np.linalg.norm(i1) * np.linalg.norm(i2)) + 1e-12))
        assert cosine >= 0.70 - 1e-6


def test_suggestion_reduces_ccs() -> None:
    request = {
        "rule1": {
            "text": "Organizations MUST collect personal data IF consent.",
            "jurisdiction": "Canada.Federal",
            "statute": "PIPEDA",
            "section": "7(3)",
            "enactment_year": 2000,
        },
        "rule2": {
            "text": "Security agencies MUST NOT collect personal data IF emergency.",
            "jurisdiction": "Canada.Federal",
            "statute": "Anti-Terrorism Act",
            "section": "83.28",
            "enactment_year": 2001,
        },
    }

    with TestClient(app, headers={"X-API-Key": "test-key"}) as client:
        analyze_resp = client.post("/v1/law/analyze", json=request)
        assert analyze_resp.status_code == 200, analyze_resp.text
        current_ccs = analyze_resp.json()["conflict_scores"]["pragmatic"]

        response = client.post("/v1/law/suggest_amendment", json=request)
        assert response.status_code == 200, response.text
        body = response.json()
    best = body["best"]
    assert best is not None
    assert best["justification"]["impact"] > 0
    assert best["estimated_ccs"] < current_ccs
