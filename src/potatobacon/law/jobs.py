from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from potatobacon.observability import bind_run_id, current_run_id, log_event, reset_run_id
from potatobacon.persistence import get_store


@dataclass
class Job:
    id: str
    type: str
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    run_id: Optional[str] = None

    def mark_running(self) -> None:
        self.status = "running"
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self, result: Dict[str, Any]) -> None:
        self.status = "completed"
        self.result = result
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, message: str) -> None:
        self.status = "failed"
        self.error = message
        self.updated_at = datetime.now(timezone.utc)


class JobManager:
    """Simple in-memory registry for background job results."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._store = get_store()

    def _add_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
        self._store.save_job(job)
        log_event("job.created", job_id=job.id, job_type=job.type)

    def create_job(self, job_type: str) -> Job:
        job = Job(id=str(uuid4()), type=job_type, run_id=current_run_id())
        self._add_job(job)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job:
            return job
        stored = self._store.load_job(job_id)
        if not stored:
            return None
        job = Job(
            id=stored["id"],
            type=stored["type"],
            status=stored["status"],
            result=stored.get("result"),
            error=stored.get("error"),
            created_at=datetime.fromisoformat(stored["created_at"]),
            updated_at=datetime.fromisoformat(stored["updated_at"]),
            run_id=stored.get("run_id"),
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def run_job(self, job: Job, runner: Callable[[], Dict[str, Any]]) -> None:
        token = bind_run_id(job.run_id)
        job.mark_running()
        self._store.save_job(job)
        log_event("job.started", job_id=job.id, job_type=job.type)
        try:
            result = runner()
        except Exception as exc:  # pragma: no cover - defensive guard
            job.mark_failed(str(exc))
            self._store.save_job(job)
            log_event("job.failed", job_id=job.id, error=str(exc))
            reset_run_id(token)
            return
        job.mark_completed(result)
        self._store.save_job(job)
        log_event("job.completed", job_id=job.id)
        reset_run_id(token)

    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()


job_manager = JobManager()

