from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass(slots=True)
class TariffScenario:
    """Scenario facts for tariff analysis."""

    name: str
    facts: Dict[str, Any]


class TariffHuntRequestModel(BaseModel):
    """Request payload for tariff arbitrage analysis."""

    law_context: Optional[str] = Field(
        default=None, description="Versioned tariff context identifier (e.g. HTS_US_2025_Q1)",
    )
    scenario: Dict[str, Any]
    mutations: Optional[Dict[str, Any]] = None
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    model_config = ConfigDict(extra="forbid")


class TariffDossierModel(BaseModel):
    """Response dossier capturing baseline and optimized tariff positions."""

    proof_id: str
    law_context: Optional[str] = None
    status: Literal["OPTIMIZED", "BASELINE"]
    baseline_duty_rate: float
    optimized_duty_rate: float
    savings_per_unit: float
    baseline_scenario: Dict[str, Any]
    optimized_scenario: Dict[str, Any]
    active_codes_baseline: List[str]
    active_codes_optimized: List[str]
    provenance_chain: List[Dict[str, Any]]
    tariff_manifest_hash: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


class TariffExplainResponseModel(BaseModel):
    """Explainability response for tariff consistency checks."""

    status: Literal["SAT", "UNSAT"]
    explanation: str
    proof_id: str
    law_context: Optional[str] = None
    unsat_core: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TariffOptimizationRequestModel(BaseModel):
    """Request payload for the tariff optimizer."""

    scenario: Dict[str, Any]
    candidate_mutations: Dict[str, List[Any]]
    law_context: Optional[str] = None
    seed: Optional[int] = None
    declared_value_per_unit: Optional[float] = 100.0
    annual_volume: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class TariffOptimizationResponseModel(BaseModel):
    """Optimizer response highlighting tariff savings."""

    status: Literal["OPTIMIZED", "BASELINE", "INFEASIBLE"]
    baseline_duty_rate: float
    optimized_duty_rate: float
    savings_per_unit: float
    best_mutation: Optional[Dict[str, Any]]
    baseline_scenario: Dict[str, Any]
    optimized_scenario: Dict[str, Any]
    active_codes_baseline: List[str]
    active_codes_optimized: List[str]
    law_context: Optional[str]
    proof_id: str
    provenance_chain: List[Dict[str, Any]]

    declared_value_per_unit: Optional[float] = None
    savings_per_unit_rate: Optional[float] = None
    savings_per_unit_value: Optional[float] = None
    annual_volume: Optional[int] = None
    annual_savings_value: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class TariffSkuOptimizationRequestModel(BaseModel):
    """SKU-level request payload for the tariff optimizer."""

    sku_id: str
    description: str
    scenario: Dict[str, Any]
    candidate_mutations: Dict[str, List[Any]]
    declared_value_per_unit: float
    annual_volume: int
    law_context: Optional[str] = None
    seed: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class TariffSkuOptimizationResponseModel(BaseModel):
    """SKU-level optimizer response with monetary savings."""

    sku_id: str
    description: str
    status: Literal["OPTIMIZED", "BASELINE", "INFEASIBLE"]
    baseline_duty_rate: float
    optimized_duty_rate: float
    savings_per_unit_rate: float
    savings_per_unit_value: float
    annual_savings_value: float
    best_mutation: Optional[Dict[str, Any]]
    baseline_scenario: Dict[str, Any]
    optimized_scenario: Dict[str, Any]
    active_codes_baseline: List[str]
    active_codes_optimized: List[str]
    law_context: Optional[str]
    proof_id: str
    provenance_chain: List[Dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


class TariffSuggestRequestModel(BaseModel):
    """Request payload for generating tariff suggestions from free text."""

    sku_id: Optional[str] = None
    description: str
    bom_text: Optional[str] = None
    declared_value_per_unit: Optional[float] = 100.0
    annual_volume: Optional[int] = None
    law_context: Optional[str] = None
    top_k: Optional[int] = 5
    seed: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class TariffSuggestionItemModel(BaseModel):
    """Single suggestion with optimized tariff projection and provenance."""

    human_summary: str
    baseline_duty_rate: float
    optimized_duty_rate: float
    savings_per_unit_rate: float
    savings_per_unit_value: float
    annual_savings_value: Optional[float]
    best_mutation: Dict[str, Any]
    active_codes_baseline: List[str]
    active_codes_optimized: List[str]
    provenance_chain: List[Dict[str, Any]]
    law_context: Optional[str]
    proof_id: str
    risk_score: Optional[int] = None
    defensibility_grade: Optional[str] = None
    risk_reasons: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")


class TariffSuggestResponseModel(BaseModel):
    """Response payload for top-k tariff suggestions."""

    status: Literal["OK", "NO_CANDIDATES"]
    sku_id: Optional[str]
    description: str
    law_context: Optional[str]
    baseline_scenario: Dict[str, Any]
    generated_candidates_count: int
    suggestions: List[TariffSuggestionItemModel]

    model_config = ConfigDict(extra="forbid")


class TariffBatchSkuModel(BaseModel):
    """SKU entry for batch tariff scanning."""

    sku_id: str
    description: str
    bom_text: Optional[str] = None
    declared_value_per_unit: Optional[float] = 100.0
    annual_volume: Optional[int] = None
    law_context: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TariffBatchScanRequestModel(BaseModel):
    """Request payload for scanning tariffs across multiple SKUs."""

    skus: List[TariffBatchSkuModel]
    top_k_per_sku: int = 3
    max_results: int = 20
    law_context: Optional[str] = None
    seed: Optional[int] = None
    include_all_suggestions: bool = False
    risk_adjusted_ranking: bool = False
    risk_penalty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Penalty multiplier applied to rank score when risk-aware ranking is enabled",
    )

    model_config = ConfigDict(extra="forbid")


class TariffBatchSkuResultModel(BaseModel):
    """Per-SKU result for a batch tariff scan."""

    sku_id: str
    description: str
    status: Literal["OK", "NO_CANDIDATES", "ERROR"]
    law_context: Optional[str]
    baseline_scenario: Dict[str, Any]
    best: Optional[TariffSuggestionItemModel] = None
    suggestions: Optional[List[TariffSuggestionItemModel]] = None
    rank_score: Optional[float] = None
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TariffBatchScanResponseModel(BaseModel):
    """Batch scan response summarizing ranked opportunities."""

    status: Literal["OK"]
    total_skus: int
    processed_skus: int
    results: List[TariffBatchSkuResultModel]
    skipped: List[TariffBatchSkuResultModel]
    generated_at: Optional[str] = None
    law_context: Optional[str] = None

    model_config = ConfigDict(extra="forbid")
