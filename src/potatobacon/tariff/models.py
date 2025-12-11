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

    model_config = ConfigDict(extra="forbid")
