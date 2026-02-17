"""BOM Upload & Batch Analysis endpoints.

Sprint C: Provides a three-step flow for real-world BOM ingestion:
  1. POST /v1/bom/upload        — Parse file, return validation summary
  2. POST /v1/bom/{upload_id}/analyze — Confirm & kick off batch analysis
  3. GET  /v1/bom/{job_id}/status     — Poll batch progress
  4. GET  /v1/bom/{job_id}/results    — Retrieve completed dossiers + summary
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, ConfigDict, Field

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, get_registry, resolve_tenant_from_request

# Sprint E: Celery integration
USE_CELERY = os.getenv("PTB_JOB_BACKEND", "threads").lower() == "celery"

if USE_CELERY:
    from potatobacon.db.models import Job as JobModel
    from potatobacon.db.session import get_db_session
    from potatobacon.workers.tasks import analyze_bom_batch
from potatobacon.tariff.bom_parser import (
    BOMParseResult,
    ParsedBOMItem,
    SkippedRow,
    compute_material_percentages,
    parse_bom_file,
)
from potatobacon.tariff.hts_hint_resolver import (
    filter_atoms_by_headings,
    resolve_hts_hint,
)
from potatobacon.api.routes_teaas import (
    TEaaSAnalyzeRequest,
    TEaaSAnalyzeResponse,
    teaas_analyze,
)
from potatobacon.tariff.duty_calculator import compute_total_duty

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/bom", tags=["bom"])

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".json", ".xlsx", ".xls"}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ParsedItemPreview(BaseModel):
    """Preview of a parsed BOM item for the validation summary."""

    row_number: int
    part_id: Optional[str] = None
    description: str
    material: Optional[str] = None
    weight_kg: Optional[float] = None
    value_usd: Optional[float] = None
    origin_country: Optional[str] = None
    hts_code: Optional[str] = None
    inferred_category: str

    model_config = ConfigDict(extra="forbid")


class SkippedRowInfo(BaseModel):
    """Info about a skipped row."""

    row_number: int
    reason: str

    model_config = ConfigDict(extra="forbid")


class BOMUploadResponse(BaseModel):
    """Response from the upload endpoint — validation summary."""

    upload_id: str
    filename: str
    status: str  # "parsed" | "error"
    total_rows: int
    parseable_rows: int
    skipped_rows: int
    skipped_reasons: List[SkippedRowInfo]
    detected_columns: List[str]
    column_mapping: Dict[str, str]
    unmatched_columns: List[str]
    material_percentages: Dict[str, Dict[str, float]]
    preview: List[ParsedItemPreview]
    warnings: List[str]

    model_config = ConfigDict(extra="forbid")


class BOMAnalyzeRequest(BaseModel):
    """Optional parameters for the analyze step."""

    law_context: Optional[str] = None
    max_mutations: int = Field(default=10, ge=1, le=50)
    import_country: Optional[str] = Field(default="US")

    model_config = ConfigDict(extra="forbid")


class BOMAnalyzeResponse(BaseModel):
    """Response from the analyze endpoint."""

    job_id: str
    upload_id: str
    status: str  # "queued"
    total_items: int
    message: str

    model_config = ConfigDict(extra="forbid")


class BOMJobStatus(BaseModel):
    """Status of a BOM batch analysis job."""

    job_id: str
    upload_id: str
    tenant_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    total: int
    completed: int
    failed: int
    submitted_at: str
    completed_at: Optional[str] = None
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SKUResult(BaseModel):
    """Per-SKU result in a batch."""

    index: int
    part_id: Optional[str] = None
    description: str
    status: str
    baseline_duty_rate: Optional[float] = None
    optimized_duty_rate: Optional[float] = None
    savings_per_unit_value: Optional[float] = None
    annual_savings_value: Optional[float] = None
    proof_id: Optional[str] = None
    proof_chain: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class PortfolioSummary(BaseModel):
    """Aggregate summary across all SKUs in a batch."""

    total_skus: int
    completed_skus: int
    failed_skus: int
    total_baseline_duty: Optional[float] = None
    total_optimized_duty: Optional[float] = None
    total_savings: Optional[float] = None
    savings_percentage: Optional[float] = None
    top_savings_skus: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BOMJobResults(BaseModel):
    """Full results for a completed BOM batch job."""

    job_id: str
    upload_id: str
    tenant_id: str
    status: str
    portfolio_summary: PortfolioSummary
    results: List[SKUResult]
    combined_proof_chain: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# In-memory stores (MVP — same pattern as routes_jobs.py)
# ---------------------------------------------------------------------------
class _UploadRecord:
    def __init__(
        self,
        upload_id: str,
        tenant_id: str,
        filename: str,
        parse_result: BOMParseResult,
    ) -> None:
        self.upload_id = upload_id
        self.tenant_id = tenant_id
        self.filename = filename
        self.parse_result = parse_result
        self.created_at = datetime.now(timezone.utc).isoformat()


class _BOMJobRecord:
    def __init__(
        self,
        job_id: str,
        upload_id: str,
        tenant_id: str,
        items: List[ParsedBOMItem],
        law_context: Optional[str],
        max_mutations: int,
        import_country: str,
    ) -> None:
        self.job_id = job_id
        self.upload_id = upload_id
        self.tenant_id = tenant_id
        self.items = items
        self.law_context = law_context
        self.max_mutations = max_mutations
        self.import_country = import_country
        self.status = "queued"
        self.total = len(items)
        self.completed = 0
        self.failed = 0
        self.submitted_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []


_uploads: Dict[str, _UploadRecord] = {}
_bom_jobs: Dict[str, _BOMJobRecord] = {}
_lock = threading.Lock()
# Z3 is not thread-safe; serialize all solver access across batch jobs
_z3_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------
def _run_bom_job(job: _BOMJobRecord, api_key: str, tenant: Tenant) -> None:
    """Execute BOM batch analysis in a background thread."""
    with _lock:
        job.status = "running"

    for i, item in enumerate(job.items):
        try:
            # Convert ParsedBOMItem to TEaaSAnalyzeRequest
            req = TEaaSAnalyzeRequest(
                description=item.description,
                origin_country=item.origin_country,
                import_country=job.import_country,
                declared_value_per_unit=item.value_usd,
                product_category=item.inferred_category,
                law_context=job.law_context,
                max_mutations=job.max_mutations,
            )

            # Z3 is not thread-safe; serialize solver access
            with _z3_lock:
                result = teaas_analyze(req, api_key=api_key, tenant=tenant)

            # Compute full duty stack using unified calculator
            duty_stack = compute_total_duty(
                base_rate=result.baseline_duty_rate,
                hts_code=item.hts_code or "",
                origin_country=item.origin_country or "",
                import_country=job.import_country or "US",
            )
            duty_breakdown = {
                "base_rate": duty_stack.base_rate,
                "base_rate_source": duty_stack.base_rate_source,
                "section_232_rate": duty_stack.section_232_rate,
                "section_301_rate": duty_stack.section_301_rate,
                "section_301_list": "",
                "section_301_citation": "",
                "section_232_citation": "",
                "ad_duty_rate": duty_stack.ad_duty_rate,
                "cvd_duty_rate": duty_stack.cvd_duty_rate,
                "ad_case_number": "",
                "fta_preference_pct": duty_stack.fta_preference_pct,
                "fta_treaty": (
                    duty_stack.fta_result.best_program.program_id
                    if duty_stack.fta_result and duty_stack.fta_result.best_program
                    else ""
                ),
                "exclusion_relief_rate": duty_stack.exclusion_relief_rate,
                "total_rate": duty_stack.total_duty_rate,
            }

            if duty_stack.adcvd_result and duty_stack.adcvd_result.order_matches:
                first_case = duty_stack.adcvd_result.order_matches[0].order.case_number
                duty_breakdown["ad_case_number"] = first_case

            # Check if manual review is required (e.g. 0% rate with no clear reason)
            status = result.status
            if result.baseline_duty_rate == 0.0 and result.baseline_effective_rate == 0.0:
                # If it's 0% but not obviously an FTA or unconditional free item, flag it
                # For now, we trust the engine, but in a real scenario we'd check for "Free" vs "0%"
                pass

            # If the engine returned a specific/compound rate that couldn't be computed
            # (usually indicated by a null rate or specific flag in metadata)
            if result.baseline_duty_rate is None:
                status = "manual_review_required"

            with _lock:
                job.results.append({
                    "index": i,
                    "part_id": item.part_id,
                    "description": item.description,
                    "origin_country": item.origin_country,
                    "hts_code": item.hts_code,
                    "value_usd": item.value_usd,
                    "annual_volume": getattr(item, 'annual_volume', None),
                    "status": status,
                    "baseline_duty_rate": result.baseline_duty_rate,
                    "baseline_effective_rate": result.baseline_effective_rate,
                    "optimized_duty_rate": result.optimized_duty_rate,
                    "optimized_effective_rate": result.optimized_effective_rate,
                    "savings_per_unit_value": result.savings_per_unit_value,
                    "annual_savings_value": result.annual_savings_value,
                    "proof_id": result.proof_id,
                    "proof_chain": result.proof_chain,
                    "duty_breakdown": duty_breakdown,
                    "optimization": {
                        "human_description": result.best_mutation.human_description,
                        "fact_patch": result.best_mutation.fact_patch,
                    } if result.best_mutation else None,
                    "errors": result.errors,
                })
                job.completed += 1

        except Exception as exc:
            with _lock:
                job.failed += 1
                job.errors.append(f"Item {i} ({item.part_id or item.description[:30]}): {exc}")
                job.results.append({
                    "index": i,
                    "part_id": item.part_id,
                    "description": item.description,
                    "status": "ERROR",
                    "errors": [str(exc)],
                })
            logger.exception("BOM job %s item %d failed", job.job_id, i)

    with _lock:
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc).isoformat()


def _compute_portfolio_summary(job: _BOMJobRecord) -> PortfolioSummary:
    """Compute aggregate portfolio summary from completed job results."""
    total_baseline = 0.0
    total_optimized = 0.0
    has_rates = False
    sku_savings: List[Dict[str, Any]] = []

    for r in job.results:
        baseline = r.get("baseline_duty_rate")
        optimized = r.get("optimized_duty_rate")
        if baseline is not None and optimized is not None:
            has_rates = True
            total_baseline += baseline
            total_optimized += optimized
            saving = baseline - optimized
            if saving > 0:
                sku_savings.append({
                    "index": r.get("index"),
                    "part_id": r.get("part_id"),
                    "description": r.get("description", "")[:60],
                    "savings": round(saving, 4),
                    "baseline_rate": baseline,
                    "optimized_rate": optimized,
                })

    # Sort by savings descending, take top 10
    sku_savings.sort(key=lambda x: -x["savings"])
    top_10 = sku_savings[:10]

    total_savings = round(total_baseline - total_optimized, 4) if has_rates else None
    savings_pct = None
    if has_rates and total_baseline > 0:
        savings_pct = round((total_baseline - total_optimized) / total_baseline * 100, 2)

    return PortfolioSummary(
        total_skus=job.total,
        completed_skus=job.completed,
        failed_skus=job.failed,
        total_baseline_duty=round(total_baseline, 4) if has_rates else None,
        total_optimized_duty=round(total_optimized, 4) if has_rates else None,
        total_savings=total_savings,
        savings_percentage=savings_pct,
        top_savings_skus=top_10,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=BOMUploadResponse)
async def upload_bom(
    file: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> BOMUploadResponse:
    """Upload and parse a BOM file (CSV, JSON, or XLSX).

    Returns a validation summary including:
    - Number of parseable/skipped rows with reasons
    - Detected column mapping
    - Preview of first 5 parsed items
    - Material percentage breakdown

    Does NOT start analysis — use POST /v1/bom/{upload_id}/analyze to confirm.
    """
    # Validate file extension
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(content)} bytes (max {MAX_FILE_SIZE})",
        )

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Parse the file
    parse_result = parse_bom_file(content, filename)

    # Check for parse errors
    if not parse_result.items and parse_result.warnings:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Failed to parse BOM file",
                "warnings": parse_result.warnings,
                "detected_columns": parse_result.detected_columns,
                "unmatched_columns": parse_result.unmatched_columns,
            },
        )

    # 422 if 0 parseable rows but no critical parse error
    if not parse_result.items:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No parseable rows found in BOM file",
                "skipped_count": len(parse_result.skipped),
                "warnings": parse_result.warnings,
            },
        )

    # Store the upload
    upload_id = f"bom_{uuid.uuid4().hex[:12]}"
    record = _UploadRecord(
        upload_id=upload_id,
        tenant_id=tenant.tenant_id,
        filename=filename,
        parse_result=parse_result,
    )
    with _lock:
        _uploads[upload_id] = record

    # Build preview (first 5 items)
    preview = [
        ParsedItemPreview(
            row_number=item.row_number,
            part_id=item.part_id,
            description=item.description[:100],
            material=item.material,
            weight_kg=item.weight_kg,
            value_usd=item.value_usd,
            origin_country=item.origin_country,
            hts_code=item.hts_code,
            inferred_category=item.inferred_category,
        )
        for item in parse_result.items[:5]
    ]

    # Compute material percentages
    mat_pcts = compute_material_percentages(parse_result.items)

    skipped_info = [
        SkippedRowInfo(row_number=s.row_number, reason=s.reason)
        for s in parse_result.skipped
    ]

    return BOMUploadResponse(
        upload_id=upload_id,
        filename=filename,
        status="parsed",
        total_rows=parse_result.total_rows,
        parseable_rows=len(parse_result.items),
        skipped_rows=len(parse_result.skipped),
        skipped_reasons=skipped_info,
        detected_columns=parse_result.detected_columns,
        column_mapping=parse_result.column_mapping,
        unmatched_columns=parse_result.unmatched_columns,
        material_percentages=mat_pcts,
        preview=preview,
        warnings=parse_result.warnings,
    )


@router.post("/{upload_id}/analyze", response_model=BOMAnalyzeResponse)
def analyze_bom(
    upload_id: str,
    req: BOMAnalyzeRequest = BOMAnalyzeRequest(),
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> BOMAnalyzeResponse:
    """Confirm and kick off batch analysis for a previously uploaded BOM.

    Each parseable row is queued as a separate analysis job. Returns a
    batch job_id that can be polled for progress.
    """
    with _lock:
        upload = _uploads.get(upload_id)

    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    if upload.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")

    items = upload.parse_result.items
    if not items:
        # Should be caught at upload, but double check
        raise HTTPException(status_code=422, detail="No parseable items in this upload")

    # Check tenant rate limit for batch size
    registry = get_registry()
    remaining = tenant.monthly_analysis_limit - tenant.monthly_analyses
    if len(items) > remaining:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Batch size ({len(items)}) exceeds remaining monthly quota ({remaining}) "
                f"for your '{tenant.plan}' plan (limit: {tenant.monthly_analysis_limit}/month). "
                f"Upgrade your plan or wait for the next billing cycle."
            ),
        )

    # Create job
    job_id = f"bomjob_{uuid.uuid4().hex[:12]}"
    job = _BOMJobRecord(
        job_id=job_id,
        upload_id=upload_id,
        tenant_id=tenant.tenant_id,
        items=items,
        law_context=req.law_context,
        max_mutations=req.max_mutations,
        import_country=req.import_country or "US",
    )

    # Sprint E: Dispatch to Celery or threading based on configuration
    if USE_CELERY:
        # Use Celery distributed task queue (production)
        from sqlalchemy.orm import Session

        db_session: Session = next(get_db_session())
        try:
            # Create job in PostgreSQL
            db_job = JobModel(
                job_id=job_id,
                tenant_id=tenant.tenant_id,
                job_type="bom_analysis",
                status="queued",
                total_items=len(items),
                input_params={
                    "upload_id": upload_id,
                    "law_context": req.law_context,
                    "max_mutations": req.max_mutations,
                    "import_country": req.import_country or "US",
                },
            )
            db_session.add(db_job)
            db_session.commit()

            # Convert items to dicts for Celery serialization
            items_dicts = [
                {
                    "part_id": item.part_id,
                    "description": item.description,
                    "origin_country": item.origin_country,
                    "value_usd": item.value_usd,
                    "inferred_category": item.inferred_category,
                }
                for item in items
            ]

            # Dispatch to Celery worker
            task = analyze_bom_batch.apply_async(
                args=[
                    job_id,
                    tenant.tenant_id,
                    items_dicts,
                    {
                        "law_context": req.law_context,
                        "max_mutations": req.max_mutations,
                        "import_country": req.import_country or "US",
                    },
                ],
                queue="bom_analysis",
            )

            # Update job with Celery task ID
            db_job.celery_task_id = task.id
            db_session.commit()

            logger.info(
                f"Dispatched BOM job {job_id} to Celery (task {task.id})"
            )
        finally:
            db_session.close()
    else:
        # Use threading (MVP/development)
        with _lock:
            _bom_jobs[job_id] = job

        thread = threading.Thread(
            target=_run_bom_job, args=(job, api_key, tenant), daemon=True
        )
        thread.start()

    return BOMAnalyzeResponse(
        job_id=job_id,
        upload_id=upload_id,
        status="queued",
        total_items=len(items),
        message=f"Batch analysis queued for {len(items)} item(s) from {upload.filename}",
    )


@router.get("/{job_id}/status", response_model=BOMJobStatus)
def bom_job_status(
    job_id: str,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> BOMJobStatus:
    """Poll batch analysis progress."""
    # Sprint E: Check PostgreSQL if Celery enabled, otherwise in-memory
    if USE_CELERY:
        from sqlalchemy.orm import Session

        db_session: Session = next(get_db_session())
        try:
            db_job = db_session.query(JobModel).filter_by(job_id=job_id).first()
            if not db_job:
                raise HTTPException(status_code=404, detail="Job not found")
            if db_job.tenant_id != tenant.tenant_id:
                raise HTTPException(status_code=403, detail="Access denied")

            return BOMJobStatus(
                job_id=db_job.job_id,
                upload_id=db_job.input_params.get("upload_id"),
                tenant_id=db_job.tenant_id,
                status=db_job.status,
                total=db_job.total_items,
                completed=db_job.completed_items,
                failed=db_job.failed_items,
                submitted_at=db_job.created_at.isoformat(),
                completed_at=db_job.completed_at.isoformat() if db_job.completed_at else None,
                errors=db_job.errors,
            )
        finally:
            db_session.close()
    else:
        # In-memory threading mode
        with _lock:
            job = _bom_jobs.get(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.tenant_id != tenant.tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")

        return BOMJobStatus(
            job_id=job.job_id,
            upload_id=job.upload_id,
            tenant_id=job.tenant_id,
            status=job.status,
            total=job.total,
            completed=job.completed,
            failed=job.failed,
            submitted_at=job.submitted_at,
            completed_at=job.completed_at,
            errors=job.errors,
        )


@router.get("/{job_id}/results", response_model=BOMJobResults)
def bom_job_results(
    job_id: str,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> BOMJobResults:
    """Retrieve completed dossiers with per-SKU savings and portfolio summary."""
    with _lock:
        job = _bom_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if job.status not in ("completed",):
        raise HTTPException(
            status_code=409,
            detail=f"Job not yet completed (status: {job.status}). "
            f"Progress: {job.completed}/{job.total}",
        )

    # Build SKU results
    sku_results = [
        SKUResult(
            index=r.get("index", 0),
            part_id=r.get("part_id"),
            description=r.get("description", ""),
            status=r.get("status", "ERROR"),
            baseline_duty_rate=r.get("baseline_duty_rate"),
            optimized_duty_rate=r.get("optimized_duty_rate"),
            savings_per_unit_value=r.get("savings_per_unit_value"),
            annual_savings_value=r.get("annual_savings_value"),
            proof_id=r.get("proof_id"),
            proof_chain=r.get("proof_chain"),
            errors=r.get("errors", []),
        )
        for r in job.results
    ]

    # Compute portfolio summary
    summary = _compute_portfolio_summary(job)

    # Build combined proof chain referencing each SKU
    combined_chain: Dict[str, Any] = {
        "type": "batch_proof_chain",
        "job_id": job.job_id,
        "upload_id": job.upload_id,
        "sku_proofs": [
            {
                "index": r.get("index"),
                "part_id": r.get("part_id"),
                "proof_id": r.get("proof_id"),
            }
            for r in job.results
            if r.get("proof_id")
        ],
        "portfolio_summary": {
            "total_baseline_duty": summary.total_baseline_duty,
            "total_optimized_duty": summary.total_optimized_duty,
            "total_savings": summary.total_savings,
        },
        "completed_at": job.completed_at,
    }

    return BOMJobResults(
        job_id=job.job_id,
        upload_id=job.upload_id,
        tenant_id=job.tenant_id,
        status=job.status,
        portfolio_summary=summary,
        results=sku_results,
        combined_proof_chain=combined_chain,
    )
