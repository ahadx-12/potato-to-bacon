"""High-level CALE engine shared by CLI and API integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .bootstrap import CALEServices, build_services
from .types import ConflictAnalysis, LegalRule


def _as_float(value: Any) -> float:
    """Return ``value`` as a plain ``float`` for JSON serialisation."""

    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Cannot coerce value {value!r} to float") from exc


@dataclass(slots=True)
class CALEEngine:
    """Facade that exposes a stable API for CLI and HTTP integrations."""

    services: CALEServices | None = None

    def __post_init__(self) -> None:
        if self.services is None:
            self.services = build_services()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _ensure_rule(self, payload: Mapping[str, Any], fallback_id: str) -> LegalRule:
        if not self.services:
            raise RuntimeError("CALE services not initialised")

        if isinstance(payload, LegalRule):
            rule = payload
        else:
            data: MutableMapping[str, Any] = dict(payload)
            text = data.get("text")
            if not text or not isinstance(text, str):
                raise ValueError("Rule payload missing 'text'")
            metadata = {
                "jurisdiction": data.get("jurisdiction", "Unknown Jurisdiction"),
                "statute": data.get("statute", "Unknown Statute"),
                "section": data.get("section", "?"),
                "enactment_year": int(data.get("enactment_year", 2000)),
                "id": data.get("id") or fallback_id,
            }
            rule = self.services.parser.parse(text, metadata)

        return self.services.feature_engine.populate(rule)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def _analysis_summary(self, analysis: ConflictAnalysis) -> dict[str, Any]:
        return {
            "conflict_intensity": _as_float(analysis.CI),
            "semantic_overlap": _as_float(analysis.K),
            "temporal_drift": _as_float(analysis.TD),
            "authority_balance": _as_float(analysis.H),
            "ccs_scores": {
                "textualist": _as_float(analysis.CCS_textualist),
                "living": _as_float(analysis.CCS_living),
                "pragmatic": _as_float(analysis.CCS_pragmatic),
            },
        }

    def _suggestion_summary(
        self,
        analysis: ConflictAnalysis,
        suggestion: Mapping[str, Any],
    ) -> dict[str, Any]:
        best = suggestion.get("best")
        if not best:
            baseline = _as_float(analysis.CCS_pragmatic)
            best = {
                "condition": "Maintain status quo pending review",
                "justification": {
                    "frequency": 0.0,
                    "semantic_relevance": 0.0,
                    "impact": 0.0,
                    "composite_score": 0.0,
                },
                "estimated_ccs": baseline,
                "suggested_text": analysis.rule1.text,
            }
            suggestions = list(suggestion.get("suggestions", []))
            suggestions.insert(0, best)
            suggestion = {
                **suggestion,
                "best": best,
                "suggestions": suggestions,
            }

        best_impact = float(best.get("justification", {}).get("impact", 0.0))
        baseline = _as_float(analysis.CCS_pragmatic)
        estimated = _as_float(best.get("estimated_ccs", baseline))
        impact = max(0.0, baseline - estimated)

        if best_impact >= 0.35:
            justification_text = "Resolves conflict consistent with higher authority"
        elif best_impact > 0.0:
            justification_text = "Improves coherence while respecting authority balance"
        else:
            justification_text = "No automated amendment identified"

        return {
            "precedent_count": int(suggestion.get("precedent_count", 0)),
            "candidates_considered": int(suggestion.get("candidates_considered", 0)),
            "suggestions": list(suggestion.get("suggestions", [])),
            "best": best,
            "suggested_amendment": {
                "condition": best.get("condition", ""),
                "impact": impact,
                "justification": justification_text,
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyse(
        self, rule1_payload: Mapping[str, Any], rule2_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not self.services:
            raise RuntimeError("CALE services not initialised")

        rule1 = self._ensure_rule(rule1_payload, "R1")
        rule2 = self._ensure_rule(rule2_payload, "R2")
        conflict = self.services.checker.check_conflict(rule1, rule2)
        analysis = self.services.calculator.compute_multiperspective(rule1, rule2, conflict)
        return self._analysis_summary(analysis)

    def suggest(
        self, rule1_payload: Mapping[str, Any], rule2_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not self.services:
            raise RuntimeError("CALE services not initialised")

        rule1 = self._ensure_rule(rule1_payload, "R1")
        rule2 = self._ensure_rule(rule2_payload, "R2")
        conflict = self.services.checker.check_conflict(rule1, rule2)
        analysis = self.services.calculator.compute_multiperspective(rule1, rule2, conflict)
        suggestion = self.services.suggester.suggest_amendment(rule1, rule2, analysis)

        result = self._analysis_summary(analysis)
        result.update(self._suggestion_summary(analysis, suggestion))
        return result
