"""Run-scoped observability helpers for CALE-LAW."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

_run_id_ctx: ContextVar[Optional[str]] = ContextVar("run_id", default=None)


def new_run_id() -> str:
    """Generate a new run identifier for correlating logs."""

    value = str(uuid.uuid4())
    _run_id_ctx.set(value)
    return value


def bind_run_id(value: Optional[str]) -> ContextVar.Token | None:
    """Bind a run_id for the current context and return the reset token."""

    if value is None:
        return None
    return _run_id_ctx.set(value)


def clear_run_id() -> None:
    """Clear the current run context."""

    _run_id_ctx.set(None)


def reset_run_id(token: Optional[ContextVar.Token]) -> None:
    """Reset the run_id context using the provided token."""

    if token is None:
        return
    _run_id_ctx.reset(token)


def current_run_id() -> Optional[str]:
    """Return the active run_id if set."""

    return _run_id_ctx.get()


def redact_api_key(raw: Optional[str]) -> str:
    """Return a redacted representation of an API key for safe logging."""

    if not raw:
        return "<missing>"
    if len(raw) <= 3:
        return "***"
    return f"{raw[:4]}***"


def log_event(message: str, **extra: object) -> None:
    """Log an event with the active run_id automatically attached."""

    payload = {"run_id": current_run_id(), **extra}
    logger.info(message, extra={"payload": payload})

