"""Redis-backed caching wrapper for Z3 solver.

Provides distributed caching and locking to replace in-memory _ATOM_CACHE
and _Z3_LOCK for horizontal scaling across multiple workers.

Usage:
    from potatobacon.law.solver_z3_cached import solve_with_redis_cache

    result = solve_with_redis_cache(
        context=context,
        facts=facts,
        atoms=atoms,
        fallback_to_local=True  # If Redis unavailable
    )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from potatobacon.caching.redis_client import get_redis_client

logger = logging.getLogger(__name__)

# Environment flag to enable/disable Redis caching
REDIS_CACHE_ENABLED = os.getenv("PTB_REDIS_CACHE", "true").lower() == "true"


def solve_with_redis_cache(
    context: Dict[str, Any],
    facts: Dict[str, Any],
    atoms: List[Any],
    fallback_to_local: bool = True,
) -> Dict[str, Any]:
    """Solve with Redis-backed caching layer.

    Args:
        context: Manifest/jurisdiction context
        facts: Input facts for Z3 solver
        atoms: Policy atoms to evaluate
        fallback_to_local: If True, use in-memory cache if Redis unavailable

    Returns:
        Solve result dict with optimal classification

    Flow:
        1. Compute cache keys from context and facts
        2. Check Redis cache
        3. On cache miss, acquire distributed lock
        4. Solve with Z3
        5. Store result in Redis
        6. Release lock
    """
    if not REDIS_CACHE_ENABLED or not fallback_to_local:
        # Redis disabled, fall back to direct Z3 solve
        return _solve_with_z3_direct(context, facts, atoms)

    redis_client = get_redis_client()

    # Check Redis connectivity
    if not redis_client.ping():
        logger.warning("Redis unavailable, falling back to local solve")
        return _solve_with_z3_direct(context, facts, atoms)

    # Compute cache keys
    context_hash = redis_client.compute_context_hash(context)
    facts_hash = redis_client.compute_facts_hash(facts)

    # Try cache first
    cached_result = redis_client.get_z3_result(context_hash, facts_hash)
    if cached_result:
        logger.info(f"Z3 cache HIT: {context_hash[:8]}:{facts_hash[:8]}")
        return cached_result

    logger.info(f"Z3 cache MISS: {context_hash[:8]}:{facts_hash[:8]}")

    # Cache miss - acquire distributed lock and solve
    lock_name = f"z3:solve:{context_hash}:{facts_hash}"

    with redis_client.distributed_lock(
        lock_name, timeout=60, blocking=True, blocking_timeout=30
    ) as acquired:
        if not acquired:
            logger.warning(
                f"Failed to acquire Z3 lock for {context_hash[:8]}:{facts_hash[:8]}, falling back to local"
            )
            return _solve_with_z3_direct(context, facts, atoms)

        # Double-check cache (another worker may have populated it)
        cached_result = redis_client.get_z3_result(context_hash, facts_hash)
        if cached_result:
            logger.info(
                f"Z3 cache HIT after lock: {context_hash[:8]}:{facts_hash[:8]}"
            )
            return cached_result

        # Solve with Z3
        result = _solve_with_z3_direct(context, facts, atoms)

        # Store in Redis (24h TTL)
        try:
            redis_client.set_z3_result(context_hash, facts_hash, result, ttl=86400)
            logger.info(
                f"Z3 result cached: {context_hash[:8]}:{facts_hash[:8]}"
            )
        except Exception as exc:
            logger.warning(f"Failed to cache Z3 result: {exc}")

        return result


def _solve_with_z3_direct(
    context: Dict[str, Any], facts: Dict[str, Any], atoms: List[Any]
) -> Dict[str, Any]:
    """Direct Z3 solve without Redis caching.

    This is the fallback implementation that uses the existing
    law/solver_z3.py functions with in-memory caching.

    Args:
        context: Manifest/jurisdiction context
        facts: Input facts
        atoms: Policy atoms

    Returns:
        Solve result dict
    """
    # Import here to avoid circular dependency
    from potatobacon.law.solver_z3 import (
        build_policy_atoms_from_rules,
        compile_atoms_to_z3,
    )
    from z3 import Optimize, sat

    # Build atoms from context (uses in-memory _ATOM_CACHE)
    manifest_hash = json.dumps(context, sort_keys=True)
    jurisdictions = tuple(context.get("jurisdictions", []))
    policy_atoms = build_policy_atoms_from_rules(manifest_hash, jurisdictions)

    if not policy_atoms:
        return {
            "status": "no_atoms",
            "message": "No policy atoms available for classification",
        }

    # Compile to Z3 (uses in-memory _Z3_LOCK)
    opt = Optimize()
    variables = {}
    compile_atoms_to_z3(policy_atoms, opt, variables)

    # Add facts as constraints
    for fact_key, fact_value in facts.items():
        if fact_key in variables:
            if fact_value:
                opt.add(variables[fact_key])
            else:
                opt.add(Not(variables[fact_key]))

    # Solve
    check_result = opt.check()

    if check_result != sat:
        return {
            "status": "unsat",
            "message": "No satisfying classification found",
        }

    # Extract model
    model = opt.model()
    result = {
        "status": "success",
        "model": {str(d): str(model[d]) for d in model.decls()},
        "atoms_evaluated": len(policy_atoms),
    }

    return result


# -------------------------------------------------------------------------
# Utility functions for migration
# -------------------------------------------------------------------------


def warm_redis_cache(contexts: List[Dict[str, Any]], facts_list: List[Dict[str, Any]]) -> int:
    """Pre-warm Redis cache with common context/facts combinations.

    Args:
        contexts: List of context dicts to warm
        facts_list: List of facts dicts to warm

    Returns:
        Number of cache entries created
    """
    redis_client = get_redis_client()
    warmed = 0

    for context in contexts:
        for facts in facts_list:
            context_hash = redis_client.compute_context_hash(context)
            facts_hash = redis_client.compute_facts_hash(facts)

            # Check if already cached
            if redis_client.get_z3_result(context_hash, facts_hash):
                continue

            # Solve and cache
            result = solve_with_redis_cache(context, facts, [], fallback_to_local=True)
            warmed += 1

    return warmed


def get_cache_stats() -> Dict[str, Any]:
    """Get Redis cache statistics.

    Returns:
        Dict with cache hit/miss counts and memory usage
    """
    redis_client = get_redis_client()

    try:
        info = redis_client.info()
        return {
            "connected": True,
            "memory_used": info.get("used_memory_human", "unknown"),
            "total_keys": info.get("db0", {}).get("keys", 0),
            "hit_rate": info.get("keyspace_hits", 0)
            / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
        }
    except Exception as exc:
        return {"connected": False, "error": str(exc)}
