from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from potatobacon.tariff.models import BaselineCandidateModel, FactEvidenceModel, StructuredBOMModel, TariffSuggestionItemModel


class SKURecordModel(BaseModel):
    """Persisted SKU profile combining description, BOM, and value inputs."""

    sku_id: str
    description: Optional[str] = None
    bom_csv: Optional[str] = None
    bom_json: StructuredBOMModel | Dict[str, Any] | None = None
    origin_country: Optional[str] = None
    export_country: Optional[str] = None
    import_country: Optional[str] = None
    declared_value_per_unit: Optional[float] = None
    annual_volume: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("bom_json", mode="before")
    @classmethod
    def _coerce_bom_json(cls, value: Any) -> Any:
        if value is None or isinstance(value, StructuredBOMModel):
            return value
        if isinstance(value, dict):
            return StructuredBOMModel(**value)
        return value

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _normalize_timestamp(cls, value: Any) -> str:
        if value is None:
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def serializable_dict(self) -> Dict[str, Any]:
        payload = self.model_dump()
        if isinstance(self.bom_json, StructuredBOMModel):
            payload["bom_json"] = self.bom_json.model_dump()
        return payload


class QuestionItemModel(BaseModel):
    """Single missing-fact question tying facts to impacted rules."""

    fact_key: str
    question: str
    why_needed: str
    accepted_evidence_types: List[str] = Field(default_factory=list)
    measurement_hint: str | None = None
    candidate_rules_affected: List[str] = Field(default_factory=list)
    lever_ids_affected: List[str] = Field(default_factory=list)
    blocks_classification: bool = False
    blocks_optimization: bool = False

    model_config = ConfigDict(extra="forbid")


class MissingFactsPackageModel(BaseModel):
    """Deterministic bundle of missing facts and follow-up questions."""

    missing_facts: List[str] = Field(default_factory=list)
    questions: List[QuestionItemModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SKUDossierBaselineModel(BaseModel):
    """Baseline classification outputs for a SKU dossier."""

    duty_rate: Optional[float] = None
    candidates: List[BaselineCandidateModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SKUDossierOptimizedModel(BaseModel):
    """Optimized suggestion, if any, for a SKU dossier."""

    suggestion: Optional[TariffSuggestionItemModel] = None

    model_config = ConfigDict(extra="forbid")


class BaselineAssignmentModel(BaseModel):
    """Concrete baseline assignment that does not rely on missing facts."""

    atom_id: Optional[str] = None
    duty_rate: Optional[float] = None
    duty_status: Optional[str] = None
    confidence: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class ConditionalPathwayModel(BaseModel):
    """Cheaper candidate pathway blocked by missing facts/evidence."""

    atom_id: str
    duty_rate: float
    missing_facts: List[str] = Field(default_factory=list)
    why_needed: List[str] = Field(default_factory=list)
    accepted_evidence_types: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TariffSkuDossierV2Model(BaseModel):
    """Unified SKU-first dossier response."""

    status: Literal[
        "OK_OPTIMIZED",
        "OK_BASELINE_ONLY",
        "INSUFFICIENT_RULE_COVERAGE",
        "INSUFFICIENT_INPUTS",
        "ERROR",
    ]
    sku_id: str
    law_context: str
    tariff_manifest_hash: str
    proof_id: Optional[str] = None
    proof_payload_hash: Optional[str] = None
    baseline: SKUDossierBaselineModel
    optimized: Optional[SKUDossierOptimizedModel] = None
    baseline_assigned: Optional[BaselineAssignmentModel] = None
    conditional_pathways: List[ConditionalPathwayModel] = Field(default_factory=list)
    questions: MissingFactsPackageModel = Field(default_factory=MissingFactsPackageModel)
    product_spec: Optional[Dict[str, Any]] = None
    compiled_facts: Optional[Dict[str, Any]] = None
    fact_evidence: Optional[List[FactEvidenceModel]] = None
    evidence_requested: bool = False
    analysis_session_id: Optional[str] = None
    attached_evidence_ids: List[str] = Field(default_factory=list)
    fact_overrides: Optional[Dict[str, FactOverrideModel]] = None
    data_quality: Optional[Dict[str, Any]] = None
    why_not_optimized: List[str] = Field(default_factory=list)
    errors: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")


class FactOverrideModel(BaseModel):
    """Session-level fact override with provenance and optional evidence links."""

    value: Any
    source: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("evidence_ids", mode="after")
    @classmethod
    def _dedupe_and_sort(cls, values: List[str]) -> List[str]:
        return sorted({str(item) for item in values})

    def serializable_dict(self) -> Dict[str, Any]:
        payload = self.model_dump()
        payload["evidence_ids"] = sorted(self.evidence_ids)
        return payload


class TariffAnalysisSessionModel(BaseModel):
    """Analysis session tying SKU inputs to iterative dossier refinement."""

    session_id: str
    sku_id: str
    law_context: str
    fact_overrides: Dict[str, FactOverrideModel] = Field(default_factory=dict)
    attached_evidence_ids: List[str] = Field(default_factory=list)
    status: Literal["OPEN", "READY_TO_OPTIMIZE", "OPTIMIZED"] = "OPEN"
    created_at: str
    updated_at: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _normalize_timestamp(cls, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @field_validator("attached_evidence_ids", mode="before")
    @classmethod
    def _normalize_attached_ids(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("attached_evidence_ids", mode="after")
    @classmethod
    def _dedupe_attached(cls, values: List[str]) -> List[str]:
        return sorted({str(item) for item in values})

    def serializable_dict(self) -> Dict[str, Any]:
        payload = self.model_dump()
        payload["attached_evidence_ids"] = sorted(self.attached_evidence_ids)
        payload["fact_overrides"] = {key: value.serializable_dict() for key, value in sorted(self.fact_overrides.items())}
        return payload


def build_sku_metadata_snapshot(
    *,
    sku_id: str | None,
    description: str | None,
    bom_json: StructuredBOMModel | Dict[str, Any] | None = None,
    bom_csv: str | None = None,
    origin_country: str | None = None,
    export_country: str | None = None,
    import_country: str | None = None,
    declared_value_per_unit: float | None = None,
    annual_volume: int | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Redacted SKU metadata suitable for persistence inside proofs."""

    if sku_id is None and description is None:
        return {}

    desc_hash = sha256((description or "").encode("utf-8")).hexdigest() if description else None
    snapshot: Dict[str, Any] = {
        "sku_id": sku_id,
        "description_hash": desc_hash,
        "has_bom_json": bool(bom_json),
        "has_bom_csv": bool(bom_csv),
        "origin_country": origin_country,
        "export_country": export_country,
        "import_country": import_country,
        "declared_value_per_unit": declared_value_per_unit,
        "annual_volume": annual_volume,
        "metadata_keys": sorted(metadata.keys()) if metadata else None,
    }
    return {key: value for key, value in snapshot.items() if value is not None}
