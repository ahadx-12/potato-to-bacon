from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List

from potatobacon.tariff.category_taxonomy import CATEGORIES


def extract_chapter(hts: str) -> int | None:
    digits = re.sub(r"\D", "", hts or "")
    if len(digits) < 2:
        return None
    return int(digits[:2])


@dataclass(frozen=True)
class CategoryScore:
    name: str
    score: float


@dataclass(frozen=True)
class CategoryResult:
    primary: CategoryScore
    confidence: float
    alternatives: List[CategoryScore]


class CategoryDetector:
    def _keyword_scores(self, description: str | None) -> Dict[str, float]:
        scores = {name: 0.0 for name in CATEGORIES.keys()}
        if not description:
            return scores
        lowered = description.lower()
        for name, spec in CATEGORIES.items():
            keywords = spec.get("keywords", [])
            hits = sum(1 for keyword in keywords if keyword in lowered)
            if hits:
                scores[name] += 1.5 * hits
        return scores

    def _chapter_scores(self, hts: str | None) -> Dict[str, float]:
        scores = {name: 0.0 for name in CATEGORIES.keys()}
        chapter = extract_chapter(hts or "")
        if chapter is None:
            return scores
        for name, spec in CATEGORIES.items():
            if chapter in spec.get("chapters", []):
                scores[name] += 2.0
        return scores

    def _value_scores(self, value: float | None) -> Dict[str, float]:
        scores = {name: 0.0 for name in CATEGORIES.keys()}
        if value is None:
            return scores
        if value >= 100.0:
            scores["machinery"] += 1.0
            scores["electronics"] += 0.5
        elif value <= 5.0:
            scores["apparel"] += 0.5
            scores["plastics"] += 0.5
            scores["furniture"] += 0.2
        else:
            scores["electronics"] += 0.4
        return scores

    def detect(self, sku: Any) -> CategoryResult:
        description = getattr(sku, "description", None)
        current_hts = getattr(sku, "current_hts", None)
        declared_value = getattr(sku, "declared_value_per_unit", None)
        unit_value = getattr(sku, "unit_value", None)
        value = declared_value if declared_value is not None else unit_value

        scores = {name: 0.0 for name in CATEGORIES.keys()}
        for source_scores in (
            self._keyword_scores(description),
            self._chapter_scores(current_hts),
            self._value_scores(value),
        ):
            for name, score in source_scores.items():
                scores[name] += score

        ordered = sorted(
            (CategoryScore(name=name, score=score) for name, score in scores.items()),
            key=lambda item: (-item.score, item.name),
        )
        primary = ordered[0]
        total = sum(score.score for score in ordered if score.score > 0)
        confidence = primary.score / total if total > 0 else 0.0
        alternatives = [score for score in ordered[1:] if score.score > 0]
        return CategoryResult(primary=primary, confidence=confidence, alternatives=alternatives)
