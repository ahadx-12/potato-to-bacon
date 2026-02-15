"""PDF download endpoints for TEaaS dossiers.

GET /v1/dossier/{sku_id}/pdf           — Per-SKU dossier PDF
GET /v1/bom/{job_id}/portfolio-pdf     — Portfolio summary PDF
GET /v1/bom/{job_id}/full-report-pdf   — Combined report PDF
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from potatobacon.api.security import require_api_key
from potatobacon.api.tenants import Tenant, resolve_tenant_from_request

logger = logging.getLogger(__name__)
router = APIRouter(tags=["pdf"])

# Cache directory for generated PDFs
_PDF_CACHE_DIR = Path(os.getenv("PTB_PDF_CACHE_DIR", "generated_pdfs"))
_PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_bom_job(job_id: str, tenant: Tenant):
    """Retrieve a BOM job from the in-memory store with tenant check."""
    from potatobacon.api.routes_bom import _bom_jobs, _lock

    with _lock:
        job = _bom_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not yet completed (status: {job.status}). "
            f"Progress: {job.completed}/{job.total}",
        )
    return job


def _find_sku_in_job(job, sku_id: str) -> Optional[Dict[str, Any]]:
    """Find a SKU result by part_id within a job."""
    for r in job.results:
        if r.get("part_id") == sku_id:
            return r
    return None


def _cache_path(key: str) -> Path:
    """Generate a cache file path for a PDF."""
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return _PDF_CACHE_DIR / f"{key}_{h}.pdf"


@router.get("/v1/dossier/{sku_id}/pdf")
def download_sku_dossier_pdf(
    sku_id: str,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    """Download per-SKU dossier PDF.

    Searches all completed jobs for this tenant to find the SKU.
    """
    from potatobacon.api.routes_bom import _bom_jobs, _lock

    # Find the SKU in any of the tenant's completed jobs
    sku_result = None
    with _lock:
        for job in _bom_jobs.values():
            if job.tenant_id == tenant.tenant_id and job.status == "completed":
                r = _find_sku_in_job(job, sku_id)
                if r:
                    sku_result = r
                    break

    if not sku_result:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{sku_id}' not found in any completed job for this tenant",
        )

    # Check cache
    cache_key = f"sku_{tenant.tenant_id}_{sku_id}"
    cached = _cache_path(cache_key)
    if cached.exists():
        return StreamingResponse(
            io.BytesIO(cached.read_bytes()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="dossier_{sku_id}.pdf"'
            },
        )

    # Generate PDF
    from potatobacon.tariff.pdf_generator import generate_sku_dossier_pdf

    pdf_bytes = generate_sku_dossier_pdf(sku_result)

    # Cache it
    try:
        cached.write_bytes(pdf_bytes)
    except OSError:
        logger.warning("Failed to cache PDF for %s", sku_id)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="dossier_{sku_id}.pdf"'},
    )


@router.get("/v1/bom/{job_id}/portfolio-pdf")
def download_portfolio_pdf(
    job_id: str,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    """Download portfolio summary PDF for a completed BOM job."""
    job = _get_bom_job(job_id, tenant)

    # Check cache
    cache_key = f"portfolio_{tenant.tenant_id}_{job_id}"
    cached = _cache_path(cache_key)
    if cached.exists():
        return StreamingResponse(
            io.BytesIO(cached.read_bytes()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="portfolio_{job_id}.pdf"'
            },
        )

    # Build portfolio data
    from potatobacon.api.routes_bom import _compute_portfolio_summary

    summary = _compute_portfolio_summary(job)
    job_data = {
        "job_id": job.job_id,
        "upload_id": job.upload_id,
        "tenant_id": job.tenant_id,
        "portfolio_summary": summary.model_dump() if hasattr(summary, "model_dump") else {
            "total_skus": summary.total_skus,
            "completed_skus": summary.completed_skus,
            "failed_skus": summary.failed_skus,
            "total_baseline_duty": summary.total_baseline_duty,
            "total_optimized_duty": summary.total_optimized_duty,
            "total_savings": summary.total_savings,
            "savings_percentage": summary.savings_percentage,
            "top_savings_skus": summary.top_savings_skus,
        },
    }

    # Run portfolio optimization if available
    portfolio_optimization = None
    try:
        from potatobacon.tariff.portfolio_optimizer import run_portfolio_optimization
        portfolio_optimization = run_portfolio_optimization(job.results)
    except Exception as exc:
        logger.warning("Portfolio optimization failed: %s", exc)

    from potatobacon.tariff.pdf_generator import generate_portfolio_pdf

    pdf_bytes = generate_portfolio_pdf(job_data, job.results, portfolio_optimization)

    # Cache
    try:
        cached.write_bytes(pdf_bytes)
    except OSError:
        logger.warning("Failed to cache portfolio PDF for %s", job_id)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="portfolio_{job_id}.pdf"'
        },
    )


@router.get("/v1/bom/{job_id}/full-report-pdf")
def download_full_report_pdf(
    job_id: str,
    api_key: str = Depends(require_api_key),
    tenant: Tenant = Depends(resolve_tenant_from_request),
):
    """Download full combined report PDF (portfolio + all SKU dossiers)."""
    job = _get_bom_job(job_id, tenant)

    # Check cache
    cache_key = f"fullreport_{tenant.tenant_id}_{job_id}"
    cached = _cache_path(cache_key)
    if cached.exists():
        return StreamingResponse(
            io.BytesIO(cached.read_bytes()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="full_report_{job_id}.pdf"'
            },
        )

    # Build job data
    from potatobacon.api.routes_bom import _compute_portfolio_summary

    summary = _compute_portfolio_summary(job)
    job_data = {
        "job_id": job.job_id,
        "upload_id": job.upload_id,
        "tenant_id": job.tenant_id,
        "portfolio_summary": summary.model_dump() if hasattr(summary, "model_dump") else {
            "total_skus": summary.total_skus,
            "completed_skus": summary.completed_skus,
            "failed_skus": summary.failed_skus,
            "total_baseline_duty": summary.total_baseline_duty,
            "total_optimized_duty": summary.total_optimized_duty,
            "total_savings": summary.total_savings,
            "savings_percentage": summary.savings_percentage,
            "top_savings_skus": summary.top_savings_skus,
        },
    }

    # Run portfolio optimization
    portfolio_optimization = None
    try:
        from potatobacon.tariff.portfolio_optimizer import run_portfolio_optimization
        portfolio_optimization = run_portfolio_optimization(job.results)
    except Exception as exc:
        logger.warning("Portfolio optimization failed: %s", exc)

    from potatobacon.tariff.pdf_generator import generate_full_report_pdf

    pdf_bytes = generate_full_report_pdf(job_data, job.results, portfolio_optimization)

    # Cache
    try:
        cached.write_bytes(pdf_bytes)
    except OSError:
        logger.warning("Failed to cache full report PDF for %s", job_id)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="full_report_{job_id}.pdf"'
        },
    )
