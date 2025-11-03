"""High-level CALE engine shared by CLI and API integrations."""

from __future__ import annotations

import math
import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

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
    # Finance-aware helpers
    # ------------------------------------------------------------------
    def _finance_weights(self) -> tuple[float, float, float, float, Mapping[str, float]]:
        config = _load_finance_config()
        alpha = float(config.get("alpha", DEFAULT_WEIGHTS["alpha"]))
        beta = float(config.get("beta", DEFAULT_WEIGHTS["beta"]))
        eta = float(config.get("eta", DEFAULT_WEIGHTS["eta"]))
        gamma = float(config.get("temporal_gamma", DEFAULT_WEIGHTS["temporal_gamma"]))
        authority = config.get("authority", DEFAULT_WEIGHTS["authority"])
        return alpha, beta, eta, gamma, authority

    def _calibration_threshold(self) -> float:
        config = _load_finance_config()
        return float(
            config.get(
                "calibration_threshold", DEFAULT_WEIGHTS["calibration_threshold"]
            )
        )

    @staticmethod
    def _semantic_overlap(rule1: LegalRule, rule2: LegalRule) -> float:
        vec1 = getattr(rule1, "interpretive_vec", None)
        vec2 = getattr(rule2, "interpretive_vec", None)
        if vec1 is None or vec2 is None:
            return 0.0
        similarity = float(np.dot(vec1, vec2))
        similarity = max(-1.0, min(1.0, similarity))
        return 0.5 * (similarity + 1.0)

    def _authority_prior(self, rule: LegalRule, authority: Mapping[str, float]) -> float:
        family = _section_family(getattr(rule, "section", ""))
        if family:
            return float(authority.get(family, 0.0))
        return 0.0

    def _compute_conflict_metrics(
        self, rule1: LegalRule, rule2: LegalRule
    ) -> tuple[float, float, float, float, float]:
        if not self.services:
            raise RuntimeError("CALE services not initialised")

        symbolic = float(self.services.checker.check_conflict(rule1, rule2))
        alpha, beta, eta, gamma, authority = self._finance_weights()

        semantic_overlap = self._semantic_overlap(rule1, rule2)
        m1 = _modality_scalar(rule1.text)
        m2 = _modality_scalar(rule2.text)
        modality_gap = abs(m1 - m2)

        bypass_flag = max(
            detects_bypass(rule1.text, rule2.text),
            detects_bypass(rule2.text, rule1.text),
        )

        if symbolic <= 0.0:
            conflict_intensity = 0.0
        else:
            conflict_intensity = _clamp01(alpha * (1.0 - semantic_overlap) + beta * modality_gap + eta * float(bypass_flag))

        temporal_drift = float(
            self.services.calculator.compute_temporal_drift(rule1, rule2)
        )

        prior1 = self._authority_prior(rule1, authority)
        prior2 = self._authority_prior(rule2, authority)
        delta_year = float(getattr(rule2, "enactment_year", 0) - getattr(rule1, "enactment_year", 0))
        authority_balance = _sigmoid((prior2 - prior1) + gamma * delta_year)

        return (
            symbolic,
            conflict_intensity,
            semantic_overlap,
            temporal_drift,
            authority_balance,
        )

    def _prepare_analysis(
        self, rule1: LegalRule, rule2: LegalRule
    ) -> tuple[ConflictAnalysis, Mapping[str, float], Mapping[str, Any]]:
        metrics = self._compute_conflict_metrics(rule1, rule2)
        (
            symbolic_conflict,
            conflict_intensity,
            semantic_overlap,
            temporal_drift,
            authority_balance,
        ) = metrics

        raw_semantic_overlap = float(semantic_overlap)
        precedent_matches: Sequence[Mapping[str, Any]] = ()
        precedent_similarity = 0.0
        if self.services and getattr(self.services, "precedent_index", None):
            precedent_matches = self.services.precedent_index.search(rule1, rule2, top_k=3)
            if precedent_matches:
                similarities = [
                    float(match.get("similarity", 0.0)) for match in precedent_matches
                ]
                precedent_similarity = float(sum(similarities) / len(similarities))
                semantic_overlap = float(
                    0.7 * float(semantic_overlap) + 0.3 * precedent_similarity
                )

        analysis = self.services.calculator.compute_multiperspective(
            rule1, rule2, conflict_intensity
        )
        analysis.CI = float(conflict_intensity)
        analysis.K = float(semantic_overlap)
        analysis.TD = float(temporal_drift)
        analysis.H = float(authority_balance)

        components = {
            "symbolic_conflict": float(symbolic_conflict),
            "contextual_similarity": float(semantic_overlap),
            "temporal_drift": float(temporal_drift),
            "authority": float(authority_balance),
        }

        calibration_threshold = self._calibration_threshold()
        calibration_confidence = _sigmoid(
            (float(conflict_intensity) - calibration_threshold) * 5.0
        )

        metadata: dict[str, Any] = {
            "precedent_matches": [dict(match) for match in precedent_matches],
            "precedent_similarity": float(precedent_similarity),
            "raw_semantic_overlap": raw_semantic_overlap,
            "calibration_confidence": float(calibration_confidence),
            "calibration_threshold": float(calibration_threshold),
        }

        return analysis, components, metadata

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def _analysis_summary(
        self,
        analysis: ConflictAnalysis,
        components: Mapping[str, float],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        scores = {
            "textualist": _as_float(analysis.CCS_textualist),
            "living": _as_float(analysis.CCS_living),
            "pragmatic": _as_float(analysis.CCS_pragmatic),
        }
        summary: dict[str, Any] = {
            "conflict_intensity": _as_float(analysis.CI),
            "semantic_overlap": _as_float(analysis.K),
            "temporal_drift": _as_float(analysis.TD),
            "authority_balance": _as_float(analysis.H),
            "components": {
                key: _as_float(value)
                for key, value in components.items()
            },
            "variance": _as_float(analysis.variance),
            "ccs_scores": scores,
            "conflict_scores": scores,
        }
        if metadata:
            if "precedent_similarity" in metadata:
                summary["precedent_similarity"] = _as_float(
                    metadata["precedent_similarity"]
                )
            if metadata.get("precedent_matches"):
                summary["precedent_matches"] = list(metadata["precedent_matches"])
            if "calibration_confidence" in metadata:
                summary["calibration_confidence"] = _as_float(
                    metadata["calibration_confidence"]
                )
        return summary

    def _suggestion_summary(
        self,
        analysis: ConflictAnalysis,
        suggestion: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
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

        precedent_context: Mapping[str, Any] | None = None
        if metadata and metadata.get("precedent_matches"):
            matches = list(metadata["precedent_matches"])
            top_match = matches[0]
            precedent_context = {
                "highlight": {
                    "id": top_match.get("id"),
                    "title": top_match.get("title"),
                    "citation": top_match.get("citation"),
                    "similarity": float(top_match.get("similarity", 0.0)),
                    "excerpt": top_match.get("excerpt", ""),
                },
                "matches": matches,
            }
            if precedent_context["highlight"].get("title"):
                justification_text = (
                    f"{justification_text}; aligned with "
                    f"{precedent_context['highlight']['title']}"
                )

        result = {
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
        if precedent_context:
            result["precedent_context"] = precedent_context
        return result

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
        analysis, components, metadata = self._prepare_analysis(rule1, rule2)
        return self._analysis_summary(analysis, components, metadata)

    def suggest(
        self, rule1_payload: Mapping[str, Any], rule2_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not self.services:
            raise RuntimeError("CALE services not initialised")

        rule1 = self._ensure_rule(rule1_payload, "R1")
        rule2 = self._ensure_rule(rule2_payload, "R2")
        analysis, components, metadata = self._prepare_analysis(rule1, rule2)
        suggestion = self.services.suggester.suggest_amendment(rule1, rule2, analysis)

        result = self._analysis_summary(analysis, components, metadata)
        result.update(self._suggestion_summary(analysis, suggestion, metadata))
        return result
FINANCE_CONFIG_PATH = Path("configs/finance.yml")

DEFAULT_WEIGHTS = {
    "alpha": 0.45,
    "beta": 0.25,
    "eta": 0.30,
    "temporal_gamma": 0.30,
    "authority": {
        "CREDIT_AGREEMENT": 2.0,
        "INDENTURE_NOTES": 1.8,
        "NOTES_TO_FS": 1.2,
        "RISK_FACTORS": 0.8,
        "LIQUIDITY": 0.6,
        "MDNA": 0.4,
    },
    "min_semantic_overlap": 0.25,
    "calibration_threshold": 0.5,
}

EXC_TERMS = (
    "unless",
    "except",
    "subject to",
    "provided that",
    "waiver",
    "amend",
    "amendment",
)
PERM_TOKENS = (
    "may",
    "allowed to",
    "permit",
    "incur",
    "borrow",
    "issue",
    "leverage",
    "indebtedness",
    "notes",
)
OBL_TOKENS = ("must", "shall")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def detects_bypass(rule_obl_text: str, rule_perm_text: str) -> int:
    t1 = rule_obl_text.lower()
    t2 = rule_perm_text.lower()
    if not any(tok in t1 for tok in OBL_TOKENS):
        return 0
    if not any(tok in t2 for tok in PERM_TOKENS):
        return 0
    if any(term in t2 for term in EXC_TERMS) or ("subject to" in t2) or ("provided that" in t2):
        return 1
    return 0


def _finance_config_module():
    spec = importlib.util.find_spec("yaml")
    if spec is None:
        return None
    return importlib.import_module("yaml")


@lru_cache(maxsize=1)
def _load_finance_config() -> Mapping[str, Any]:
    yaml_module = _finance_config_module()
    if yaml_module is None or not FINANCE_CONFIG_PATH.exists():
        return DEFAULT_WEIGHTS
    try:
        with FINANCE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml_module.safe_load(handle) or {}
    except Exception:
        return DEFAULT_WEIGHTS
    weights = data.get("weights", {})
    signals = data.get("signals", {})
    authority = dict(DEFAULT_WEIGHTS["authority"])
    authority.update(weights.get("authority", {}))
    return {
        "alpha": float(weights.get("alpha", DEFAULT_WEIGHTS["alpha"])),
        "beta": float(weights.get("beta", DEFAULT_WEIGHTS["beta"])),
        "eta": float(weights.get("eta", DEFAULT_WEIGHTS["eta"])),
        "temporal_gamma": float(
            weights.get("temporal_gamma", DEFAULT_WEIGHTS["temporal_gamma"])
        ),
        "authority": authority,
        "min_semantic_overlap": float(
            signals.get(
                "min_semantic_overlap", DEFAULT_WEIGHTS["min_semantic_overlap"]
            )
        ),
        "calibration_threshold": float(
            signals.get(
                "calibration_threshold", DEFAULT_WEIGHTS["calibration_threshold"]
            )
        ),
    }


def _section_family(section: str | None) -> str | None:
    if not section:
        return None
    sec = section.upper()
    if "CREDIT" in sec or "FACILITIES" in sec:
        return "CREDIT_AGREEMENT"
    if "INDENTURE" in sec or "NOTE" in sec:
        return "INDENTURE_NOTES"
    if "NOTES" in sec and "FINANC" in sec:
        return "NOTES_TO_FS"
    if "RISK" in sec and "FACTOR" in sec:
        return "RISK_FACTORS"
    if "LIQUIDITY" in sec:
        return "LIQUIDITY"
    if "MANAGEMENT" in sec or "MD&A" in sec:
        return "MDNA"
    return None


def _modality_scalar(text: str) -> float:
    lowered = text.lower()
    if "must not" in lowered or "shall not" in lowered:
        return -1.0
    if "may not" in lowered:
        return -0.3
    if "must" in lowered or "shall" in lowered:
        return 1.0
    if "may" in lowered:
        return 0.3
    return 0.0
