"""TEaaS orchestration endpoint.

POST /v1/teaas/analyze — Single endpoint that:
  1. Accepts product description + BOM (structured or free-text)
  2. Compiles facts via the fact compiler
  3. Runs baseline HTS classification via Z3
  4. Discovers optimization mutations via the MutationEngine
  5. Tests each mutation and ranks savings
  6. Returns a full dossier with cryptographic proof

This replaces the need to call /parse, /analyze, /optimize separately.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, get_registry, resolve_tenant_from_request
from potatobacon.law.solver_z3 import analyze_scenario
from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.proofs.proof_chain import ProofChain
from potatobacon.tariff.atom_utils import atom_provenance
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.engine import apply_mutations, compute_duty_result
from potatobacon.tariff.fact_compiler import compile_facts
from potatobacon.tariff.models import TariffScenario
from potatobacon.tariff.mutation_engine import MutationEngine
from potatobacon.tariff.mutation_generator import (
    generate_candidate_mutations,
    generate_mutation_candidates,
    infer_product_profile,
)
from potatobacon.tariff.overlays import effective_duty_rate, evaluate_overlays
from potatobacon.tariff.origin_engine import build_origin_policy_atoms
from potatobacon.tariff.product_schema import ProductCategory, ProductSpecModel
from potatobacon.tariff.models import StructuredBOMModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/teaas", tags=["teaas"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class TEaaSAnalyzeRequest(BaseModel):
    """Input for the unified TEaaS analysis pipeline."""

    description: str = Field(..., min_length=3, description="Product description")
    bom_text: Optional[str] = Field(default=None, description="Free-text BOM")
    bom_json: Optional[StructuredBOMModel] = Field(default=None, description="Structured BOM")
    origin_country: Optional[str] = Field(default=None, description="Country of origin (ISO)")
    export_country: Optional[str] = Field(default=None, description="Export country (ISO)")
    import_country: Optional[str] = Field(default="US", description="Import country (ISO)")
    declared_value_per_unit: Optional[float] = Field(default=None, ge=0.0)
    annual_volume: Optional[int] = Field(default=None, ge=0)
    product_category: Optional[str] = Field(
        default=None, description="Override inferred product category"
    )
    law_context: Optional[str] = Field(
        default=None, description="Tariff context (e.g. HTS_US_2025_SLICE)"
    )
    max_mutations: int = Field(default=10, ge=1, le=50)

    model_config = ConfigDict(extra="forbid")


class TEaaSMutationResult(BaseModel):
    """Result of testing a single mutation."""

    mutation_id: int
    human_description: str
    fact_patch: Dict[str, Any]
    projected_duty_rate: float
    effective_duty_rate: float
    savings_vs_baseline: float
    effective_savings: float
    verified: bool = True

    model_config = ConfigDict(extra="forbid")


class TEaaSAnalyzeResponse(BaseModel):
    """Full TEaaS analysis dossier."""

    status: str  # "OPTIMIZED" | "BASELINE" | "NO_DUTY_RULE" | "ERROR"
    tenant_id: Optional[str] = None
    description: str
    law_context: str
    tariff_manifest_hash: str

    # Classification
    inferred_category: Optional[str] = None
    compiled_facts: Dict[str, Any]

    # Baseline
    baseline_duty_rate: Optional[float] = None
    baseline_effective_rate: Optional[float] = None
    baseline_active_codes: List[str] = Field(default_factory=list)

    # Optimization
    mutations_tested: int = 0
    best_mutation: Optional[TEaaSMutationResult] = None
    all_mutations: List[TEaaSMutationResult] = Field(default_factory=list)

    # Optimized position (after best mutation)
    optimized_duty_rate: Optional[float] = None
    optimized_effective_rate: Optional[float] = None
    optimized_active_codes: List[str] = Field(default_factory=list)

    # Savings
    duty_savings_pct: Optional[float] = None
    effective_savings_pct: Optional[float] = None
    savings_per_unit_value: Optional[float] = None
    annual_savings_value: Optional[float] = None

    # Proof
    proof_id: Optional[str] = None
    proof_payload_hash: Optional[str] = None
    proof_chain: Optional[Dict[str, Any]] = None

    # Provenance
    provenance_chain: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    analyzed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CATEGORY_KEYWORDS: Dict[str, ProductCategory] = {
    "shoe": ProductCategory.FOOTWEAR,
    "sneaker": ProductCategory.FOOTWEAR,
    "footwear": ProductCategory.FOOTWEAR,
    "boot": ProductCategory.FOOTWEAR,
    "bolt": ProductCategory.FASTENER,
    "screw": ProductCategory.FASTENER,
    "fastener": ProductCategory.FASTENER,
    "nut": ProductCategory.FASTENER,
    "washer": ProductCategory.FASTENER,
    "pcb": ProductCategory.ELECTRONICS,
    "circuit": ProductCategory.ELECTRONICS,
    "battery": ProductCategory.ELECTRONICS,
    "cable": ProductCategory.ELECTRONICS,
    "router": ProductCategory.ELECTRONICS,
    "motor": ProductCategory.ELECTRONICS,
    "electronics": ProductCategory.ELECTRONICS,
    "wiring": ProductCategory.ELECTRONICS,
    "harness": ProductCategory.ELECTRONICS,
    "shirt": ProductCategory.APPAREL_TEXTILE,
    "garment": ProductCategory.APPAREL_TEXTILE,
    "apparel": ProductCategory.APPAREL_TEXTILE,
    "textile": ProductCategory.TEXTILE,
    "fabric": ProductCategory.TEXTILE,
    "cotton": ProductCategory.TEXTILE,
    "polyester": ProductCategory.TEXTILE,
    "chair": ProductCategory.FURNITURE,
    "desk": ProductCategory.FURNITURE,
    "furniture": ProductCategory.FURNITURE,
    "sofa": ProductCategory.FURNITURE,
    "table": ProductCategory.FURNITURE,
    "seat": ProductCategory.FURNITURE,
}


def _infer_category(description: str) -> ProductCategory:
    """Keyword-based category inference from product description."""
    lower = description.lower()
    for keyword, category in _CATEGORY_KEYWORDS.items():
        if keyword in lower:
            return category
    return ProductCategory.OTHER


_MATERIAL_KEYWORDS: List[str] = [
    "steel", "aluminum", "textile", "rubber", "leather",
    "plastic", "synthetic", "copper", "iron", "wood",
    "polyethylene", "nylon", "cotton", "polyester",
]


def _extract_materials_from_text(text: str) -> List[Dict[str, str]]:
    """Extract material hints from free-text description."""
    lower = text.lower()
    found: List[Dict[str, str]] = []
    for mat in _MATERIAL_KEYWORDS:
        if mat in lower:
            found.append({"component": "body", "material": mat})
    return found


def _build_product_spec(req: TEaaSAnalyzeRequest) -> ProductSpecModel:
    """Build a ProductSpecModel from the TEaaS request."""
    if req.product_category:
        try:
            category = ProductCategory(req.product_category)
        except ValueError:
            category = _infer_category(req.description)
    else:
        category = _infer_category(req.description)

    materials: List[Dict[str, str]] = []
    if req.bom_json:
        for item in req.bom_json.items:
            if item.material:
                materials.append({
                    "component": item.description or item.part_id or "component",
                    "material": item.material,
                })

    # Fall back to text-based material extraction
    if not materials:
        combined = " ".join(filter(None, [req.description, req.bom_text or ""]))
        materials = _extract_materials_from_text(combined)

    return ProductSpecModel(
        product_category=category,
        materials=[
            {"component": m["component"], "material": m["material"]}
            for m in materials
        ],
        origin_country=req.origin_country,
        export_country=req.export_country,
        import_country=req.import_country or "US",
        declared_value_per_unit=req.declared_value_per_unit,
        annual_volume=req.annual_volume,
    )


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=TEaaSAnalyzeResponse)
def teaas_analyze(
    req: TEaaSAnalyzeRequest,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> TEaaSAnalyzeResponse:
    """Unified TEaaS analysis: classify, optimize, prove."""

    errors: List[str] = []
    resolved_context = req.law_context or DEFAULT_CONTEXT_ID

    # Load atoms
    try:
        atoms, context_meta = load_atoms_for_context(resolved_context)
    except (KeyError, ValueError) as exc:
        return TEaaSAnalyzeResponse(
            status="ERROR",
            tenant_id=tenant.tenant_id,
            description=req.description,
            law_context=resolved_context,
            tariff_manifest_hash="",
            compiled_facts={},
            errors=[f"Failed to load context: {exc}"],
        )

    context_id = context_meta["context_id"]
    manifest_hash = context_meta["manifest_hash"]
    duty_rates = context_meta.get("duty_rates") or {}

    # Build product spec and compile facts
    product_spec = _build_product_spec(req)
    facts, evidence = compile_facts(product_spec)

    # Baseline classification
    baseline = TariffScenario(name="baseline", facts=deepcopy(facts))
    origin_atoms = build_origin_policy_atoms()
    combined_atoms = list(origin_atoms) + list(atoms)

    baseline_result = compute_duty_result(atoms, baseline, duty_rates=duty_rates)
    baseline_rate = baseline_result.duty_rate if baseline_result.duty_rate is not None else 0.0

    baseline_overlays = evaluate_overlays(
        facts=baseline.facts,
        active_codes=[a.source_id for a in baseline_result.active_atoms],
        origin_country=req.origin_country,
        import_country=req.import_country or "US",
    )
    baseline_effective = effective_duty_rate(baseline_rate, baseline_overlays)

    baseline_codes = sorted([a.source_id for a in baseline_result.active_atoms])

    if baseline_result.status not in ("OK",):
        # Still try mutations but note the issue
        if baseline_result.status == "NO_DUTY_RULE_ACTIVE":
            errors.append("No duty rule matched baseline facts")
        elif baseline_result.status == "UNSAT":
            errors.append("Baseline scenario is logically inconsistent")

    # Discover mutations via the Z3-driven engine
    mutation_engine = MutationEngine(atoms, duty_rates=duty_rates)
    derived_mutations = mutation_engine.discover_mutations(
        baseline, baseline_rate, max_candidates=req.max_mutations
    )

    # Also try legacy hardcoded mutations
    legacy_mutations = generate_candidate_mutations(
        infer_product_profile(req.description, req.bom_text)
    )

    # Test all mutations
    mutation_results: List[TEaaSMutationResult] = []
    mid = 0

    for dm in derived_mutations:
        mid += 1
        mutated = TariffScenario(
            name=f"mutation-{mid}",
            facts={**deepcopy(facts), **dm.fact_patch},
        )
        mut_result = compute_duty_result(atoms, mutated, duty_rates=duty_rates)
        mut_rate = mut_result.duty_rate if mut_result.duty_rate is not None else baseline_rate
        mut_overlays = evaluate_overlays(
            facts=mutated.facts,
            active_codes=[a.source_id for a in mut_result.active_atoms],
            origin_country=req.origin_country,
            import_country=req.import_country or "US",
        )
        mut_effective = effective_duty_rate(mut_rate, mut_overlays)
        mutation_results.append(TEaaSMutationResult(
            mutation_id=mid,
            human_description=dm.human_description,
            fact_patch=dm.fact_patch,
            projected_duty_rate=mut_rate,
            effective_duty_rate=mut_effective,
            savings_vs_baseline=baseline_rate - mut_rate,
            effective_savings=baseline_effective - mut_effective,
            verified=dm.verified,
        ))

    for legacy_mut in legacy_mutations:
        if isinstance(legacy_mut, dict):
            patch = legacy_mut
            desc = "Legacy optimization candidate"
        else:
            continue
        mid += 1
        mutated = apply_mutations(baseline, patch)
        mut_result = compute_duty_result(atoms, mutated, duty_rates=duty_rates)
        mut_rate = mut_result.duty_rate if mut_result.duty_rate is not None else baseline_rate
        mut_overlays = evaluate_overlays(
            facts=mutated.facts,
            active_codes=[a.source_id for a in mut_result.active_atoms],
            origin_country=req.origin_country,
            import_country=req.import_country or "US",
        )
        mut_effective = effective_duty_rate(mut_rate, mut_overlays)
        if mut_effective < baseline_effective:
            mutation_results.append(TEaaSMutationResult(
                mutation_id=mid,
                human_description=desc,
                fact_patch=patch,
                projected_duty_rate=mut_rate,
                effective_duty_rate=mut_effective,
                savings_vs_baseline=baseline_rate - mut_rate,
                effective_savings=baseline_effective - mut_effective,
                verified=True,
            ))

    # Rank by effective savings
    mutation_results.sort(key=lambda m: -m.effective_savings)
    best = mutation_results[0] if mutation_results else None

    # Compute optimized position
    if best and best.effective_savings > 0:
        optimized_rate = best.projected_duty_rate
        optimized_effective = best.effective_duty_rate
        optimized_scenario = TariffScenario(
            name="optimized",
            facts={**deepcopy(facts), **best.fact_patch},
        )
        opt_result = compute_duty_result(atoms, optimized_scenario, duty_rates=duty_rates)
        optimized_codes = sorted([a.source_id for a in opt_result.active_atoms])
        status = "OPTIMIZED"
    else:
        optimized_rate = baseline_rate
        optimized_effective = baseline_effective
        optimized_scenario = baseline
        optimized_codes = baseline_codes
        status = "BASELINE" if baseline_result.status == "OK" else "NO_DUTY_RULE"

    # Build proof chain (per-step hashing)
    chain = ProofChain()
    chain.add_step("bom_input", input_data=req.model_dump(), output_data=req.model_dump())
    chain.add_step("fact_compilation", input_data=product_spec.model_dump(), output_data=facts)
    chain.add_step("z3_baseline", input_data=facts, output_data={
        "status": baseline_result.status,
        "duty_rate": baseline_rate,
        "active_codes": baseline_codes,
    })
    chain.add_step("mutation_discovery", input_data={"baseline_rate": baseline_rate}, output_data={
        "mutations_tested": len(mutation_results),
        "best_savings": best.effective_savings if best else 0,
    })
    chain.add_step("classification", input_data={
        "optimized_rate": optimized_rate,
        "optimized_codes": optimized_codes,
    }, output_data={
        "status": status,
        "effective_rate": optimized_effective,
    })

    # Build provenance
    sat_b, active_b, unsat_b = analyze_scenario(baseline.facts, combined_atoms)
    sat_o, active_o, unsat_o = analyze_scenario(optimized_scenario.facts, combined_atoms)
    provenance: List[Dict[str, Any]] = []
    for atom in active_b:
        if atom.source_id in duty_rates:
            provenance.append(atom_provenance(atom, "baseline"))
    for atom in active_o:
        if atom.source_id in duty_rates:
            provenance.append(atom_provenance(atom, "optimized"))

    chain.add_step("dossier", input_data={"provenance_count": len(provenance)}, output_data={
        "baseline_effective": baseline_effective,
        "optimized_effective": optimized_effective,
    })

    # Record proof (with tenant_id)
    proof_handle = record_tariff_proof(
        law_context=context_id,
        base_facts=baseline.facts,
        mutations=best.fact_patch if best else None,
        baseline_active=active_b,
        optimized_active=active_o,
        baseline_sat=sat_b,
        optimized_sat=sat_o,
        baseline_duty_rate=baseline_rate,
        optimized_duty_rate=optimized_rate,
        baseline_duty_status=baseline_result.status,
        optimized_duty_status="OK",
        baseline_scenario=baseline.facts,
        optimized_scenario=optimized_scenario.facts,
        baseline_unsat_core=unsat_b,
        optimized_unsat_core=unsat_o,
        provenance_chain=provenance,
        tariff_manifest_hash=manifest_hash,
    )

    # Increment tenant usage
    get_registry().increment_usage(tenant.tenant_id)

    # Compute savings values
    duty_savings_pct = baseline_rate - optimized_rate if baseline_rate else None
    effective_savings_pct = baseline_effective - optimized_effective if baseline_effective else None
    savings_per_unit_value = None
    annual_savings_value = None
    if req.declared_value_per_unit and effective_savings_pct:
        savings_per_unit_value = round(req.declared_value_per_unit * effective_savings_pct / 100, 2)
        if req.annual_volume:
            annual_savings_value = round(savings_per_unit_value * req.annual_volume, 2)

    return TEaaSAnalyzeResponse(
        status=status,
        tenant_id=tenant.tenant_id,
        description=req.description,
        law_context=context_id,
        tariff_manifest_hash=manifest_hash,
        inferred_category=product_spec.product_category.value,
        compiled_facts=facts,
        baseline_duty_rate=baseline_rate,
        baseline_effective_rate=baseline_effective,
        baseline_active_codes=baseline_codes,
        mutations_tested=len(mutation_results),
        best_mutation=best,
        all_mutations=mutation_results,
        optimized_duty_rate=optimized_rate,
        optimized_effective_rate=optimized_effective,
        optimized_active_codes=optimized_codes,
        duty_savings_pct=duty_savings_pct,
        effective_savings_pct=effective_savings_pct,
        savings_per_unit_value=savings_per_unit_value,
        annual_savings_value=annual_savings_value,
        proof_id=proof_handle.proof_id,
        proof_payload_hash=proof_handle.proof_payload_hash,
        proof_chain=chain.to_dict(),
        provenance_chain=provenance,
        errors=errors if errors else [],
    )
