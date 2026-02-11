"""Celery workers for distributed task processing.

Replaces threading-based job execution with production-ready Celery workers.
"""

from potatobacon.workers.celery_app import celery_app
from potatobacon.workers.tasks import analyze_bom_batch, portfolio_scan

__all__ = ["celery_app", "analyze_bom_batch", "portfolio_scan"]
