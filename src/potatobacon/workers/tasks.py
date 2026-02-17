"""Celery tasks for BOM analysis, portfolio scanning, etc.

Replaces threading.Thread-based job execution from api/routes_bom.py
with distributed Celery tasks for production scalability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

from potatobacon.api.routes_teaas import TEaaSAnalyzeRequest, teaas_analyze
from potatobacon.db.models import Job, Tenant
from potatobacon.db.session import get_standalone_session
from potatobacon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="potatobacon.workers.tasks.analyze_bom_batch")
def analyze_bom_batch(
    self,
    job_id: str,
    tenant_id: str,
    items: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze a batch of BOM items (replaces _run_bom_job thread).

    Args:
        self: Celery task instance (for progress updates)
        job_id: Job identifier in PostgreSQL
        tenant_id: Tenant identifier
        items: List of ParsedBOMItem dicts
        params: Analysis parameters (law_context, import_country, max_mutations)

    Returns:
        Dict with job_id and completion summary
    """
    with get_standalone_session() as session:
        # Update job status to running
        job = session.query(Job).filter_by(job_id=job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = "running"
        job.celery_task_id = self.request.id
        session.commit()

        # Get tenant for context
        tenant = session.query(Tenant).filter_by(tenant_id=tenant_id).first()
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        results = []
        errors = []

        # Process each item
        for i, item in enumerate(items):
            try:
                # Convert ParsedBOMItem to TEaaSAnalyzeRequest
                req = TEaaSAnalyzeRequest(
                    description=item.get("description", ""),
                    origin_country=item.get("origin_country"),
                    import_country=params.get("import_country", "US"),
                    declared_value_per_unit=item.get("value_usd"),
                    hts_hint=item.get("hts_code"),
                    product_category=item.get("inferred_category"),
                    law_context=params.get("law_context"),
                    max_mutations=params.get("max_mutations"),
                )

                # Run analysis (Z3 distributed lock handled in solver_z3_cached)
                result = teaas_analyze(req, tenant=tenant)

                results.append(
                    {
                        "index": i,
                        "part_id": item.get("part_id"),
                        "description": item.get("description", "")[:100],
                        "status": result.status,
                        "baseline_duty_rate": result.baseline_duty_rate,
                        "optimized_duty_rate": result.optimized_duty_rate,
                        "savings_per_unit_value": result.savings_per_unit_value,
                        "proof_id": result.proof_id,
                    }
                )

                job.completed_items += 1

            except Exception as exc:
                logger.error(f"BOM item {i} failed for job {job_id}: {exc}")
                job.failed_items += 1
                errors.append(f"Item {i}: {str(exc)}")

            # Update progress and persist to DB
            if (i + 1) % 5 == 0 or i == len(items) - 1:
                session.commit()

                # Update Celery task progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "completed": job.completed_items,
                        "total": job.total_items,
                        "failed": job.failed_items,
                    },
                )

        # Mark job as completed
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.result = {"results": results}
        job.errors = errors
        session.commit()

        # Fire webhook if configured
        if job.callback_url:
            _fire_webhook(job.callback_url, job_id, "completed", results)

        logger.info(
            f"BOM job {job_id} completed: {len(results)} succeeded, {len(errors)} failed"
        )

        return {
            "job_id": job_id,
            "completed": len(results),
            "failed": len(errors),
        }


@celery_app.task(name="potatobacon.workers.tasks.portfolio_scan")
def portfolio_scan(tenant_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Scan tenant's SKU portfolio for duty optimization opportunities.

    Args:
        tenant_id: Tenant identifier
        params: Scan parameters

    Returns:
        Dict with savings opportunities
    """
    # TODO: Implement portfolio scanning logic
    # This would iterate through all SKUs for a tenant and run analysis
    logger.info(f"Portfolio scan for tenant {tenant_id} (not implemented)")
    return {"status": "not_implemented"}


def _fire_webhook(url: str, job_id: str, status: str, results: List[Dict]) -> None:
    """Send webhook notification on job completion.

    Args:
        url: Webhook URL
        job_id: Job identifier
        status: Job status
        results: Analysis results
    """
    try:
        response = httpx.post(
            url,
            json={
                "job_id": job_id,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_items": len(results),
                    "completed": len([r for r in results if r.get("status") == "success"]),
                },
            },
            timeout=10,
        )
        response.raise_for_status()
        logger.info(f"Webhook fired for job {job_id}: {url}")
    except Exception as exc:
        logger.warning(f"Webhook failed for job {job_id}: {exc}")
