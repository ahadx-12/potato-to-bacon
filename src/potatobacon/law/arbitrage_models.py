"""Pydantic schemas for the arbitrage engine responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ArbitrageScenario(BaseModel):
    jurisdictions: List[str] = Field(default_factory=list)
    facts: Dict[str, Any] = Field(default_factory=dict)


class ArbitrageMetrics(BaseModel):
    value: float
    entropy: float
    kappa: float
    risk: float
    contradiction_probability: float
    score: float
    value_components: Optional[Dict[str, float]] = None
    risk_components: Optional[Dict[str, float]] = None
    score_components: Optional[Dict[str, float]] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    seed: Optional[int] = None


class ProvenanceStep(BaseModel):
    step: int
    jurisdiction: str
    rule_id: str
    type: str
    role: str
    summary: Optional[str] = None
    atom_id: Optional[str] = None
    urn: Optional[str] = None
    citations: Optional[List[str]] = None
    effective_date: Optional[str] = None


class DependencyNode(BaseModel):
    id: str
    jurisdiction: str
    label: str
    urn: Optional[str] = None
    citations: Optional[List[str]] = None


class DependencyEdge(BaseModel):
    from_id: str
    to_id: str
    relation: str


class DependencyGraph(BaseModel):
    nodes: List[DependencyNode]
    edges: List[DependencyEdge] = Field(default_factory=list)


class ArbitrageCandidateModel(BaseModel):
    scenario: Dict[str, Any]
    metrics: ArbitrageMetrics
    proof_trace: List[str]


class ArbitrageDossierModel(BaseModel):
    golden_scenario: ArbitrageScenario
    metrics: ArbitrageMetrics
    proof_trace: List[str]
    risk_flags: List[str]
    candidates: List[ArbitrageCandidateModel]
    provenance_chain: List[ProvenanceStep]
    dependency_graph: Optional[DependencyGraph] = None
    engine_version: Optional[str] = None
    manifest_hash: Optional[str] = None
