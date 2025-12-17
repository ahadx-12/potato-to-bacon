from __future__ import annotations

from fastapi import HTTPException

from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, available_context_ids


def unknown_law_context_error(context_id: str | None, domain: str = "tariff") -> HTTPException:
    """Return a standardized HTTP 400 error for unknown law contexts."""

    attempted = context_id if context_id is not None else DEFAULT_CONTEXT_ID
    return HTTPException(
        status_code=400,
        detail={
            "message": f"Unknown law_context: {attempted}",
            "available_contexts": available_context_ids(domain=domain),
        },
    )
