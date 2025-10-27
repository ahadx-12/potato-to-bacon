"""High-level CALE engine shared by CLI and API integrations."""

from __future__ import annotations

import math
import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

from . import finance_extract
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
    def _finance_settings(self) -> Mapping[str, Any]:
        return _load_finance_config()

    def _finance_enabled(self) -> bool:
        settings = self._finance_settings()
        return bool(settings.get("enable", False))

    def _finance_weights(self) -> tuple[float, float, float, float, Mapping[str, float]]:
        settings = self._finance_settings()
        weights = settings.get("weights", {}) if isinstance(settings, Mapping) else {}
        alpha = float(weights.get("alpha", DEFAULT_FINANCE_CONFIG["weights"]["alpha"]))
        beta = float(weights.get("beta", DEFAULT_FINANCE_CONFIG["weights"]["beta"]))
        eta = float(weights.get("eta", DEFAULT_FINANCE_CONFIG["weights"]["eta"]))
        gamma = float(weights.get("temporal_gamma", DEFAULT_FINANCE_CONFIG["weights"]["temporal_gamma"]))
        authority = dict(DEFAULT_FINANCE_CONFIG["weights"]["authority"])
        authority.update(weights.get("authority", {}))
        return alpha, beta, eta, gamma, authority

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
    ) -> tuple[float, float, float, float]:
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

        return conflict_intensity, semantic_overlap, temporal_drift, authority_balance

    def _prepare_analysis(self, rule1: LegalRule, rule2: LegalRule) -> ConflictAnalysis:
        metrics = self._compute_conflict_metrics(rule1, rule2)
        conflict_intensity, semantic_overlap, temporal_drift, authority_balance = metrics
        analysis = self.services.calculator.compute_multiperspective(
            rule1, rule2, conflict_intensity
        )
        analysis.CI = float(conflict_intensity)
        analysis.K = float(semantic_overlap)
        analysis.TD = float(temporal_drift)
        analysis.H = float(authority_balance)
        return analysis

    def analyse_finance(
        self,
        doc_text: str,
        *,
        prior_doc_text: str | None = None,
        section_hint: str | None = None,
        strict: bool | None = None,
    ) -> dict[str, Any]:
        """Run the finance-specific extractor on raw filing text."""

        strict_mode = bool(strict)
        result = finance_extract.analyse_finance_sections(
            doc_text,
            prior_text=prior_doc_text,
            section_hint=section_hint,
            strict=strict_mode,
        )
        result.setdefault("temporal_drift", 0.0)
        result.setdefault("ccs_scores", {})
        result.setdefault("suggested_amendment", {
            "condition": "Maintain status quo pending finance review",
            "impact": 0.0,
            "justification": "Finance module placeholder",
        })
        return result

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
        analysis = self._prepare_analysis(rule1, rule2)
        summary = self._analysis_summary(analysis)
        if self._finance_enabled():
            section_hint = str(getattr(rule1, "section", "")) or str(getattr(rule2, "section", ""))
            summary["finance"] = self.analyse_finance(
                f"{rule1.text}\n{rule2.text}",
                section_hint=section_hint or None,
            )
        return summary

    def suggest(
        self, rule1_payload: Mapping[str, Any], rule2_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not self.services:
            raise RuntimeError("CALE services not initialised")

        rule1 = self._ensure_rule(rule1_payload, "R1")
        rule2 = self._ensure_rule(rule2_payload, "R2")
        analysis = self._prepare_analysis(rule1, rule2)
        suggestion = self.services.suggester.suggest_amendment(rule1, rule2, analysis)

        result = self._analysis_summary(analysis)
        result.update(self._suggestion_summary(analysis, suggestion))
        if self._finance_enabled():
            section_hint = str(getattr(rule1, "section", "")) or str(getattr(rule2, "section", ""))
            result["finance"] = self.analyse_finance(
                f"{rule1.text}\n{rule2.text}",
                section_hint=section_hint or None,
            )
        return result
FINANCE_CONFIG_PATH = Path("configs/finance.yml")

DEFAULT_FINANCE_CONFIG = {
    "enable": False,
    "lambda_bypass": 0.5,
    "weights": {
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
    },
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
    base: dict[str, Any] = {
        "enable": DEFAULT_FINANCE_CONFIG.get("enable", False),
        "lambda_bypass": DEFAULT_FINANCE_CONFIG.get("lambda_bypass", 0.5),
        "weights": dict(DEFAULT_FINANCE_CONFIG["weights"]),
    }
    if yaml_module is None or not FINANCE_CONFIG_PATH.exists():
        return base
    try:
        with FINANCE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml_module.safe_load(handle) or {}
    except Exception:
        return base
    if isinstance(data, Mapping):
        base["enable"] = bool(data.get("enable", base["enable"]))
        if "lambda_bypass" in data:
            try:
                base["lambda_bypass"] = float(data["lambda_bypass"])
            except Exception:
                pass
        weights = dict(base["weights"])
        incoming = data.get("weights", {}) if isinstance(data.get("weights"), Mapping) else {}
        for key in ("alpha", "beta", "eta", "temporal_gamma"):
            if key in incoming:
                try:
                    weights[key] = float(incoming[key])
                except Exception:
                    continue
        authority = dict(weights.get("authority", {}))
        if isinstance(incoming.get("authority"), Mapping):
            authority.update(incoming["authority"])
        weights["authority"] = authority
        base["weights"] = weights
        for key, value in data.items():
            if key not in ("weights", "enable", "lambda_bypass"):
                base[key] = value
    return base


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
