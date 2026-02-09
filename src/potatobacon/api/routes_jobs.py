"""Async BOM analysis job queue.

Manages the lifecycle of tariff analysis jobs:
  POST /v1/jobs/analyze  → Submit BOM, get job_id
  GET  /v1/jobs/{job_id} → Poll for completion
  GET  /v1/jobs          → List tenant's jobs

Z3 solving isn't instant for complex BOMs. This module provides
async job management so batch uploads (e.g. 500 SKUs) don't block.
For MVP we use an in-memory queue with thread-based execution;
production would swap to Celery/Redis.
"""

from __future__ import annotations

import logging
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, get_registry, resolve_tenant_from_request
from potatobacon.api.routes_teaas import (
    TEaaSAnalyzeRequest,
    TEaaSAnalyzeResponse,
    teaas_analyze,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Job models
# ---------------------------------------------------------------------------
class JobSubmitRequest(BaseModel):
    """Submit one or more BOMs for async analysis."""

    items: List[TEaaSAnalyzeRequest] = Field(
        ..., min_length=1, max_length=500, description="BOMs to analyze"
    )

    model_config = ConfigDict(extra="forbid")


class JobStatus(BaseModel):
    """Status of a single analysis job."""

    job_id: str
    tenant_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    total_items: int
    completed_items: int
    failed_items: int
    submitted_at: str
    completed_at: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class JobSubmitResponse(BaseModel):
    """Response after submitting a job."""

    job_id: str
    status: str
    total_items: int
    message: str

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# In-memory job store (MVP — replace with Redis/Celery for production)
# ---------------------------------------------------------------------------
class _JobRecord:
    def __init__(
        self, job_id: str, tenant_id: str, items: List[TEaaSAnalyzeRequest]
    ) -> None:
        self.job_id = job_id
        self.tenant_id = tenant_id
        self.items = items
        self.status = "queued"
        self.total_items = len(items)
        self.completed_items = 0
        self.failed_items = 0
        self.submitted_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []


_jobs: Dict[str, _JobRecord] = {}
_lock = threading.Lock()


def _run_job(job: _JobRecord, api_key: str, tenant: Tenant) -> None:
    """Execute a job in a background thread."""
    with _lock:
        job.status = "running"

    for i, item in enumerate(job.items):
        try:
            result = teaas_analyze(item, api_key=api_key, tenant=tenant)
            with _lock:
                job.results.append({
                    "index": i,
                    "description": item.description,
                    "status": result.status,
                    "baseline_duty_rate": result.baseline_duty_rate,
                    "optimized_duty_rate": result.optimized_duty_rate,
                    "savings_per_unit_value": result.savings_per_unit_value,
                    "annual_savings_value": result.annual_savings_value,
                    "proof_id": result.proof_id,
                })
                job.completed_items += 1
        except Exception as exc:
            with _lock:
                job.failed_items += 1
                job.errors.append(f"Item {i}: {exc}")
            logger.exception("Job %s item %d failed", job.job_id, i)

    with _lock:
        job.status = "completed" if job.failed_items == 0 else "completed"
        job.completed_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=JobSubmitResponse)
def submit_job(
    req: JobSubmitRequest,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> JobSubmitResponse:
    """Submit a batch of BOMs for async analysis."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = _JobRecord(job_id=job_id, tenant_id=tenant.tenant_id, items=req.items)

    with _lock:
        _jobs[job_id] = job

    # Launch in background thread
    thread = threading.Thread(
        target=_run_job, args=(job, api_key, tenant), daemon=True
    )
    thread.start()

    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        total_items=len(req.items),
        message=f"Analysis queued for {len(req.items)} item(s)",
    )


@router.get("/{job_id}", response_model=JobStatus)
def get_job(
    job_id: str,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
) -> JobStatus:
    """Check the status of an analysis job."""
    with _lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return JobStatus(
        job_id=job.job_id,
        tenant_id=job.tenant_id,
        status=job.status,
        total_items=job.total_items,
        completed_items=job.completed_items,
        failed_items=job.failed_items,
        submitted_at=job.submitted_at,
        completed_at=job.completed_at,
        results=job.results if job.status == "completed" else None,
        errors=job.errors,
    )


@router.get("", response_model=List[JobStatus])
def list_jobs(
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
    limit: int = Query(default=20, ge=1, le=100),
) -> List[JobStatus]:
    """List jobs for the authenticated tenant."""
    with _lock:
        tenant_jobs = [
            j for j in _jobs.values() if j.tenant_id == tenant.tenant_id
        ]

    # Sort by submission time descending
    tenant_jobs.sort(key=lambda j: j.submitted_at, reverse=True)
    return [
        JobStatus(
            job_id=j.job_id,
            tenant_id=j.tenant_id,
            status=j.status,
            total_items=j.total_items,
            completed_items=j.completed_items,
            failed_items=j.failed_items,
            submitted_at=j.submitted_at,
            completed_at=j.completed_at,
            results=None,  # Don't include results in list view
            errors=j.errors,
        )
        for j in tenant_jobs[:limit]
    ]
