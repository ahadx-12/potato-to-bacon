from __future__ import annotations

import os
import threading
import time
from typing import Dict, Iterable, Optional, Tuple

from fastapi import Header, HTTPException, Request


ENGINE_VERSION = os.getenv("CALE_ENGINE_VERSION", "0.2.0")


def _parse_api_keys(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {key.strip() for key in raw.split(",") if key.strip()}


def allowed_api_keys() -> set[str]:
    """Return the configured API keys from env or fallback to a dev key."""

    keys = _parse_api_keys(os.getenv("CALE_API_KEYS"))
    if not keys:
        keys = {"dev-key"}
    return keys


class RateLimiter:
    """Lightweight in-process rate limiter keyed by (api_key, route)."""

    def __init__(self, rate_per_minute: int = 60, window_seconds: int | None = None) -> None:
        self.rate_per_minute = max(1, rate_per_minute)
        self.window_seconds = max(1, window_seconds or int(os.getenv("CALE_RATE_WINDOW_SEC", "60")))
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, str], Tuple[int, int]] = {}

    def _current_window(self) -> int:
        env_window = os.getenv("CALE_RATE_WINDOW_SEC")
        if env_window:
            try:
                self.window_seconds = max(1, int(env_window))
            except ValueError:
                self.window_seconds = max(1, self.window_seconds)
        return int(time.time() // self.window_seconds)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()

    def check(self, api_key: str, route: str) -> None:
        window = self._current_window()
        key = (api_key, route)
        with self._lock:
            count, active_window = self._counters.get(key, (0, window))
            if active_window != window:
                count = 0
                active_window = window

            if count >= self.rate_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": "Rate limit exceeded",
                        "limit_per_minute": self.rate_per_minute,
                        "route": route,
                    },
                )

            self._counters[key] = (count + 1, active_window)


rate_limiter = RateLimiter(
    rate_per_minute=int(os.getenv("CALE_RATE_LIMIT_PER_MINUTE", "60"))
)


def require_api_key(
    request: Request, x_api_key: Optional[str] = Header(None)
) -> str:
    """Validate the provided API key and enforce per-route rate limits."""

    keys = allowed_api_keys()
    if not x_api_key:
        raise HTTPException(status_code=401, detail={"message": "Missing API key"})
    if x_api_key not in keys:
        raise HTTPException(status_code=401, detail={"message": "Invalid API key"})

    route = request.url.path
    rate_limiter.check(x_api_key, route)
    return x_api_key


def set_rate_limit(limit: int) -> None:
    """Utility hook for tests to reconfigure the limiter."""

    global rate_limiter
    rate_limiter = RateLimiter(
        rate_per_minute=max(1, int(limit)),
        window_seconds=int(os.getenv("CALE_RATE_WINDOW_SEC", "60")),
    )

