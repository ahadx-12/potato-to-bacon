from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from uuid import uuid4


@dataclass
class Job:
    id: str
    type: str
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

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

    def _add_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def create_job(self, job_type: str) -> Job:
        job = Job(id=str(uuid4()), type=job_type)
        self._add_job(job)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def run_job(self, job: Job, runner: Callable[[], Dict[str, Any]]) -> None:
        job.mark_running()
        try:
            result = runner()
        except Exception as exc:  # pragma: no cover - defensive guard
            job.mark_failed(str(exc))
            return
        job.mark_completed(result)

    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()


job_manager = JobManager()

