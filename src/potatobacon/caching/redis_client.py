"""Redis caching layer for Z3 solve results and USITC lookups.

Provides distributed caching and locking to replace in-memory caches
and enable horizontal scaling across multiple workers.

Cache Keys:
- z3:solve:{context_hash}:{facts_hash} → Serialized Z3 result (TTL: 24h)
- usitc:chapter:{chapter_num} → JSONL atoms for chapter (TTL: 7d)
- z3:lock:{resource_id} → Distributed lock (TTL: 30s)
- tenant:{tenant_id}:usage → Monthly analysis counter
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import redis
from redis.lock import Lock


class RedisClient:
    """Redis client for caching and distributed locks.

    Replaces:
    - In-memory _ATOM_CACHE in law/solver_z3.py
    - In-memory _Z3_LOCK with distributed lock
    - Tenant usage counters from tenants.py
    """

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = redis.from_url(
            self.url,
            decode_responses=True,  # Auto-decode strings
            socket_keepalive=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

    # -------------------------------------------------------------------------
    # Z3 Solver Caching
    # -------------------------------------------------------------------------

    def get_z3_result(
        self, context_hash: str, facts_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached Z3 solve result.

        Args:
            context_hash: SHA-256 hash of manifest/jurisdiction context
            facts_hash: SHA-256 hash of input facts

        Returns:
            Cached result dict or None if cache miss
        """
        key = f"z3:solve:{context_hash}:{facts_hash}"
        data = self._client.get(key)
        if data:
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def set_z3_result(
        self,
        context_hash: str,
        facts_hash: str,
        result: Dict[str, Any],
        ttl: int = 86400,
    ) -> None:
        """Cache Z3 solve result with 24h TTL.

        Args:
            context_hash: SHA-256 hash of manifest/jurisdiction context
            facts_hash: SHA-256 hash of input facts
            result: Solve result to cache
            ttl: Time-to-live in seconds (default: 24 hours)
        """
        key = f"z3:solve:{context_hash}:{facts_hash}"
        self._client.setex(key, ttl, json.dumps(result))

    @staticmethod
    def compute_context_hash(context: Dict[str, Any]) -> str:
        """Compute stable hash of context for cache key."""
        canonical = json.dumps(context, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def compute_facts_hash(facts: Dict[str, Any]) -> str:
        """Compute stable hash of facts for cache key."""
        canonical = json.dumps(facts, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # USITC Chapter Caching
    # -------------------------------------------------------------------------

    def get_usitc_atoms(self, chapter: str) -> Optional[list]:
        """Retrieve cached USITC atoms for chapter.

        Args:
            chapter: HTS chapter number (e.g., "84", "90")

        Returns:
            List of PolicyAtom dicts or None if cache miss
        """
        key = f"usitc:chapter:{chapter}"
        data = self._client.get(key)
        if data:
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def set_usitc_atoms(
        self, chapter: str, atoms: list, ttl: int = 604800
    ) -> None:
        """Cache USITC atoms with 7d TTL.

        Args:
            chapter: HTS chapter number
            atoms: List of PolicyAtom dicts to cache
            ttl: Time-to-live in seconds (default: 7 days)
        """
        key = f"usitc:chapter:{chapter}"
        self._client.setex(key, ttl, json.dumps(atoms))

    # -------------------------------------------------------------------------
    # Distributed Locking
    # -------------------------------------------------------------------------

    @contextmanager
    def distributed_lock(
        self,
        lock_name: str,
        timeout: int = 30,
        blocking: bool = True,
        blocking_timeout: Optional[float] = None,
    ):
        """Distributed lock to replace _Z3_LOCK for multi-worker deployment.

        Usage:
            redis_client = get_redis_client()
            with redis_client.distributed_lock('z3:solve'):
                result = solve_with_z3(...)

        Args:
            lock_name: Unique lock identifier
            timeout: Lock expiration in seconds (auto-release)
            blocking: If True, wait for lock; if False, fail immediately
            blocking_timeout: Max seconds to wait for lock (None = infinite)

        Yields:
            True if lock acquired, False if not
        """
        lock = Lock(
            self._client,
            lock_name,
            timeout=timeout,
            blocking=blocking,
            blocking_timeout=blocking_timeout,
        )
        acquired = False
        try:
            acquired = lock.acquire()
            yield acquired
        finally:
            if acquired:
                try:
                    lock.release()
                except redis.exceptions.LockError:
                    # Lock already expired/released
                    pass

    # -------------------------------------------------------------------------
    # Tenant Usage Tracking
    # -------------------------------------------------------------------------

    def increment_tenant_usage(self, tenant_id: str) -> int:
        """Atomically increment monthly analysis counter.

        Args:
            tenant_id: Tenant identifier

        Returns:
            New usage count after increment
        """
        key = f"tenant:{tenant_id}:usage"
        return self._client.incr(key)

    def get_tenant_usage(self, tenant_id: str) -> int:
        """Get current monthly analysis count.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Current usage count (0 if not set)
        """
        key = f"tenant:{tenant_id}:usage"
        count = self._client.get(key)
        return int(count) if count else 0

    def reset_tenant_usage(self, tenant_id: str) -> None:
        """Reset monthly counter (called by Stripe webhook on billing cycle).

        Args:
            tenant_id: Tenant identifier
        """
        key = f"tenant:{tenant_id}:usage"
        self._client.delete(key)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def ping(self) -> bool:
        """Check Redis connectivity.

        Returns:
            True if Redis is reachable, False otherwise
        """
        try:
            return self._client.ping()
        except redis.exceptions.ConnectionError:
            return False

    def info(self) -> Dict[str, Any]:
        """Get Redis server info.

        Returns:
            Dict with server stats (version, memory, clients, etc.)
        """
        return self._client.info()


# -------------------------------------------------------------------------
# Singleton instance
# -------------------------------------------------------------------------

_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Get singleton Redis client.

    Returns:
        Initialized RedisClient instance
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
