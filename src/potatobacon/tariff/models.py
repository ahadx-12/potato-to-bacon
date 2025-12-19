from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from potatobacon.tariff.product_schema import ProductSpecModel

if TYPE_CHECKING:
    from potatobacon.tariff.product_schema import ProductSpecModel


class TextEvidenceModel(BaseModel):
    """Span-level evidence extracted from product text inputs."""

    source: Literal["description", "bom_text", "bom_json", "payload"]
    snippet: str
    start: int | None = None
    end: int | None = None

    model_config = ConfigDict(extra="forbid")


class FactEvidenceModel(BaseModel):
    """Evidence attached to a derived tariff fact."""

    fact_key: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[TextEvidenceModel] = Field(default_factory=list)
    derived_from: List[str] = Field(default_factory=list)
    risk_reason: Optional[str] = Field(default=None)

    model_config = ConfigDict(extra="forbid")


class BOMLineItemModel(BaseModel):
    """Structured Bill of Materials line item."""

    part_id: Optional[str] = None
    description: str
    material: Optional[str] = None
    quantity: Optional[float] = None
    unit_cost: Optional[float] = None
    country_of_origin: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class StructuredBOMModel(BaseModel):
    """Normalized BOM payload for deterministic ingestion."""

    items: List[BOMLineItemModel]
    currency: Optional[str] = Field(default="USD")

    model_config = ConfigDict(extra="forbid")


@dataclass(slots=True)
class TariffScenario:
    """Scenario facts for tariff analysis."""

    name: str
    facts: Dict[str, Any]


class TariffHuntRequestModel(BaseModel):
    """Request payload for tariff arbitrage analysis."""

    law_context: Optional[str] = Field(
        default=None, description="Versioned tariff context identifier (e.g. HTS_US_DEMO_2025)",
    )
    scenario: Dict[str, Any]
    mutations: Optional[Dict[str, Any]] = None
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    origin_country: Optional[str] = None
    export_country: Optional[str] = None
    import_country: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TariffDossierModel(BaseModel):
    """Response dossier capturing baseline and optimized tariff positions."""

    proof_id: str
    proof_payload_hash: str
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


class BaselineCandidateModel(BaseModel):
    """Single baseline classification/duty outcome with provenance."""

    candidate_id: str
    active_codes: List[str]
    duty_rate: float
    provenance_chain: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    missing_facts: List[str] = Field(default_factory=list)
    compliance_flags: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class TariffSkuDossierModel(BaseModel):
    """Unified dossier for any tariff SKU evaluation."""

    status: Literal[
        "OK_OPTIMIZED",
        "OK_BASELINE_ONLY",
        "INSUFFICIENT_RULE_COVERAGE",
        "INSUFFICIENT_INPUTS",
        "ERROR",
    ]
    sku_id: Optional[str]
    law_context: str
    tariff_manifest_hash: str
    proof_id: Optional[str] = None
    proof_payload_hash: Optional[str] = None
    product_spec: Optional[Dict[str, Any]] = None
    compiled_facts: Optional[Dict[str, Any]] = None
    fact_evidence: Optional[List[Any]] = None
    baseline_candidates: List[BaselineCandidateModel] = Field(default_factory=list)
    best_optimization: Optional[Dict[str, Any]] = None
    why_not_optimized: List[str] = Field(default_factory=list)
    errors: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")


class TariffExplainResponseModel(BaseModel):
    """Explainability response for tariff consistency checks."""

    status: Literal["SAT", "UNSAT"]
    explanation: str
    proof_id: str
    proof_payload_hash: str
    law_context: Optional[str] = None
    unsat_core: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_flags: Dict[str, bool] = Field(default_factory=dict)
    recommended_next_inputs: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TariffOptimizationRequestModel(BaseModel):
    """Request payload for the tariff optimizer."""

    scenario: Dict[str, Any]
    candidate_mutations: Dict[str, List[Any]]
    law_context: Optional[str] = None
    seed: Optional[int] = None
    declared_value_per_unit: Optional[float] = 100.0
    annual_volume: Optional[int] = None
    origin_country: Optional[str] = None
    export_country: Optional[str] = None
    import_country: Optional[str] = None

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
    tariff_manifest_hash: Optional[str] = None
    proof_id: str
    proof_payload_hash: str
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
    tariff_manifest_hash: Optional[str] = None
    proof_id: str
    proof_payload_hash: str
    provenance_chain: List[Dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


class TariffSuggestRequestModel(BaseModel):
    """Request payload for generating tariff suggestions from free text."""

    sku_id: Optional[str] = None
    description: str
    bom_text: Optional[str] = None
    bom_json: Optional[StructuredBOMModel] = None
    bom_csv: Optional[str] = None
    declared_value_per_unit: Optional[float] = 100.0
    annual_volume: Optional[int] = None
    law_context: Optional[str] = None
    top_k: Optional[int] = 5
    seed: Optional[int] = None
    include_fact_evidence: bool = False
    origin_country: Optional[str] = None
    export_country: Optional[str] = None
    import_country: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TariffSuggestionItemModel(BaseModel):
    """Single suggestion with optimized tariff projection and provenance."""

    human_summary: str
    lever_id: Optional[str] = None
    lever_feasibility: Optional[str] = None
    evidence_requirements: List[str] = Field(default_factory=list)
    baseline_duty_rate: float
    optimized_duty_rate: float
    savings_per_unit_rate: float
    savings_per_unit_value: float
    annual_savings_value: Optional[float]
    best_mutation: Dict[str, Any]
    classification_confidence: Optional[float] = None
    active_codes_baseline: List[str]
    active_codes_optimized: List[str]
    provenance_chain: List[Dict[str, Any]]
    law_context: Optional[str]
    proof_id: str
    proof_payload_hash: str
    risk_score: Optional[int] = None
    defensibility_grade: Optional[str] = None
    risk_reasons: Optional[List[str]] = None
    tariff_manifest_hash: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TariffSuggestResponseModel(BaseModel):
    """Response payload for top-k tariff suggestions."""

    status: Literal[
        "OK_OPTIMIZED",
        "OK_BASELINE_ONLY",
        "INSUFFICIENT_RULE_COVERAGE",
        "INSUFFICIENT_INPUTS",
        "ERROR",
    ]
    sku_id: Optional[str]
    description: str
    law_context: Optional[str]
    baseline_scenario: Dict[str, Any]
    generated_candidates_count: int
    suggestions: List[TariffSuggestionItemModel]
    tariff_manifest_hash: Optional[str] = None
    fact_evidence: Optional[List[FactEvidenceModel]] = None
    product_spec: Optional["ProductSpecModel"] = None
    baseline_candidates: List[BaselineCandidateModel] = Field(default_factory=list)
    why_not_optimized: List[str] = Field(default_factory=list)
    proof_id: Optional[str] = None
    proof_payload_hash: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TariffParseRequestModel(BaseModel):
    """Request payload for deterministic tariff parsing."""

    sku_id: Optional[str] = None
    description: str
    bom_text: Optional[str] = None
    bom_json: Optional[StructuredBOMModel] = None
    bom_csv: Optional[str] = None
    origin_country: Optional[str] = None
    export_country: Optional[str] = None
    import_country: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TariffParseResponseModel(BaseModel):
    """Response payload for tariff parsing with compiled facts and evidence."""

    sku_id: Optional[str] = None
    product_spec: "ProductSpecModel"
    compiled_facts: Dict[str, Any]
    fact_evidence: List[FactEvidenceModel]
    extraction_evidence: List[TextEvidenceModel]

    model_config = ConfigDict(extra="forbid")


TariffSuggestResponseModel.model_rebuild()
TariffParseResponseModel.model_rebuild()


class TariffBatchSkuModel(BaseModel):
    """SKU entry for batch tariff scanning."""

    sku_id: str
    description: str
    bom_text: Optional[str] = None
    bom_json: Optional[StructuredBOMModel] = None
    bom_csv: Optional[str] = None
    declared_value_per_unit: Optional[float] = 100.0
    annual_volume: Optional[int] = None
    law_context: Optional[str] = None
    origin_country: Optional[str] = None
    export_country: Optional[str] = None
    import_country: Optional[str] = None

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
    status: Literal[
        "OK_OPTIMIZED",
        "OK_BASELINE_ONLY",
        "INSUFFICIENT_RULE_COVERAGE",
        "INSUFFICIENT_INPUTS",
        "ERROR",
    ]
    law_context: Optional[str]
    tariff_manifest_hash: Optional[str] = None
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
    tariff_manifest_hash: Optional[str] = None

    model_config = ConfigDict(extra="forbid")
