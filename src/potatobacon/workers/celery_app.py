"""Celery app configuration for async job processing.

Replaces thread-based job queue from api/routes_jobs.py and law/jobs.py
with distributed Celery workers for production scalability.

Broker: Redis
Result Backend: PostgreSQL (reuses same DB connection)
Worker Concurrency: 4 (Z3 solver lock limits parallelism)
"""

from __future__ import annotations

import os

from celery import Celery

# Celery configuration from environment
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND = os.getenv(
    "CELERY_RESULT_BACKEND",
    "db+postgresql://postgres:postgres@localhost:5432/potatobacon",
)

# Create Celery app
celery_app = Celery(
    "potatobacon",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["potatobacon.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,
    # Timeouts
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit
    # Worker settings
    worker_prefetch_multiplier=1,  # No prefetch for long-running jobs
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevent memory leaks)
    worker_disable_rate_limits=True,
    # Result backend settings
    result_backend_transport_options={
        "master_name": "mymaster",
    },
    result_expires=3600,  # Keep results for 1 hour
    # Broker settings
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    # Task routing
    task_routes={
        "potatobacon.workers.tasks.analyze_bom_batch": {"queue": "bom_analysis"},
        "potatobacon.workers.tasks.portfolio_scan": {"queue": "portfolio"},
    },
    # Beat schedule (for periodic tasks)
    beat_schedule={
        # Example: Check for tariff schedule changes daily
        # "check-tariff-updates": {
        #     "task": "potatobacon.workers.tasks.check_tariff_updates",
        #     "schedule": crontab(hour=0, minute=0),  # Daily at midnight
        # },
    },
)


# Worker initialization hook
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Configure periodic tasks (if needed)."""
    pass


if __name__ == "__main__":
    celery_app.start()
