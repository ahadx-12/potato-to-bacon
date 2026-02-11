"""Caching layer for TEaaS production infrastructure.

Provides Redis-based distributed caching and locking.
"""

from potatobacon.caching.redis_client import RedisClient, get_redis_client

__all__ = ["RedisClient", "get_redis_client"]
