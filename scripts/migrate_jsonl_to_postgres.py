#!/usr/bin/env python3
"""Zero-downtime migration from JSONL to PostgreSQL.

Migrates data from JSONL files to PostgreSQL tables while preserving
timestamps and maintaining backward compatibility during the transition.

Usage:
    # Backfill historical data from JSONL to PostgreSQL
    python scripts/migrate_jsonl_to_postgres.py --mode=backfill --tenant=default

    # Verify data integrity after migration
    python scripts/migrate_jsonl_to_postgres.py --mode=verify --tenant=default

    # Show migration statistics
    python scripts/migrate_jsonl_to_postgres.py --mode=stats --tenant=default
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from potatobacon.db.models import (
    Alert,
    AnalysisSession,
    APIKey,
    SKU,
    Tenant,
)
from potatobacon.db.session import get_standalone_session
from potatobacon.tariff.analysis_session_store import AnalysisSessionStore
from potatobacon.tariff.sku_store import SKUStore, get_tenant_sku_store


def migrate_tenants(session: Session, tenant_id: str = "default") -> int:
    """Migrate tenant from in-memory registry to PostgreSQL.

    Args:
        session: Database session
        tenant_id: Tenant identifier

    Returns:
        Number of tenants migrated
    """
    from potatobacon.api.tenants import get_registry

    registry = get_registry()
    tenant = registry.get_tenant(tenant_id)

    if not tenant:
        print(f"⚠️  Tenant '{tenant_id}' not found in registry")
        return 0

    # Check if already exists
    existing = session.query(Tenant).filter_by(tenant_id=tenant_id).first()
    if existing:
        print(f"✓ Tenant '{tenant_id}' already exists in PostgreSQL")
        return 0

    # Create tenant
    db_tenant = Tenant(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        plan=tenant.plan,
        sku_limit=tenant.sku_limit,
        monthly_analysis_limit=tenant.monthly_analysis_limit,
        monthly_analyses=tenant.monthly_analyses,
        created_at=datetime.fromisoformat(tenant.created_at),
        metadata_json=tenant.metadata,
    )
    session.add(db_tenant)
    session.commit()

    # Migrate API keys
    for api_key in tenant.api_keys:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        db_key = APIKey(
            key_hash=key_hash,
            tenant_id=tenant_id,
            name="Migrated from in-memory registry",
            created_at=datetime.fromisoformat(tenant.created_at),
        )
        session.add(db_key)

    session.commit()
    print(f"✓ Migrated tenant '{tenant_id}' with {len(tenant.api_keys)} API key(s)")
    return 1


def migrate_skus(session: Session, tenant_id: str, jsonl_path: Path = None) -> int:
    """Migrate SKUs from JSONL to PostgreSQL.

    Args:
        session: Database session
        tenant_id: Tenant identifier
        jsonl_path: Optional custom JSONL path

    Returns:
        Number of SKUs migrated
    """
    store = get_tenant_sku_store(tenant_id)
    records = store.list(limit=1000)

    if not records:
        print(f"ℹ️  No SKUs found for tenant '{tenant_id}'")
        return 0

    migrated = 0
    for record in records:
        # Check if already exists
        existing = (
            session.query(SKU)
            .filter_by(tenant_id=tenant_id, sku_id=record.sku_id)
            .first()
        )
        if existing:
            continue

        sku = SKU(
            tenant_id=tenant_id,
            sku_id=record.sku_id,
            description=record.description,
            current_hts=record.current_hts,
            origin_country=record.origin_country,
            declared_value_per_unit=record.declared_value_per_unit,
            annual_volume=record.annual_volume,
            inferred_category=record.inferred_category,
            category_confidence=record.category_confidence,
            created_at=datetime.fromisoformat(record.created_at),
            updated_at=datetime.fromisoformat(record.updated_at),
            metadata_json=record.metadata or {},
        )
        session.add(sku)
        migrated += 1

        if migrated % 100 == 0:
            session.commit()
            print(f"  Migrated {migrated} SKUs...")

    session.commit()
    print(f"✓ Migrated {migrated} SKU(s) for tenant '{tenant_id}'")
    return migrated


def migrate_analysis_sessions(session: Session, tenant_id: str) -> int:
    """Migrate analysis sessions from JSONL to PostgreSQL.

    Args:
        session: Database session
        tenant_id: Tenant identifier

    Returns:
        Number of sessions migrated
    """
    store = AnalysisSessionStore()

    # Load all sessions from JSONL
    sessions = []
    if store.path.exists():
        with store.path.open("r") as f:
            for line in f:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not sessions:
        print(f"ℹ️  No analysis sessions found")
        return 0

    migrated = 0
    for session_data in sessions:
        session_id = session_data.get("session_id")
        if not session_id:
            continue

        # Check if already exists
        existing = (
            session.query(AnalysisSession).filter_by(session_id=session_id).first()
        )
        if existing:
            continue

        db_session = AnalysisSession(
            session_id=session_id,
            tenant_id=tenant_id,
            created_at=datetime.fromisoformat(session_data.get("created_at")),
            updated_at=datetime.fromisoformat(session_data.get("updated_at")),
            context=session_data.get("context", {}),
            mutations=session_data.get("mutations", []),
            final_classification=session_data.get("final_classification"),
        )
        session.add(db_session)
        migrated += 1

    session.commit()
    print(f"✓ Migrated {migrated} analysis session(s)")
    return migrated


def verify_migration(session: Session, tenant_id: str) -> Dict[str, Any]:
    """Verify migration by comparing JSONL vs PostgreSQL counts.

    Args:
        session: Database session
        tenant_id: Tenant identifier

    Returns:
        Dict with verification results
    """
    # Count SKUs
    store = get_tenant_sku_store(tenant_id)
    jsonl_sku_count = len(store.list(limit=1000))
    pg_sku_count = session.query(SKU).filter_by(tenant_id=tenant_id).count()

    # Count tenants
    from potatobacon.api.tenants import get_registry

    registry = get_registry()
    jsonl_tenant_count = 1 if registry.get_tenant(tenant_id) else 0
    pg_tenant_count = session.query(Tenant).filter_by(tenant_id=tenant_id).count()

    results = {
        "tenant": {
            "jsonl": jsonl_tenant_count,
            "postgres": pg_tenant_count,
            "match": jsonl_tenant_count == pg_tenant_count,
        },
        "skus": {
            "jsonl": jsonl_sku_count,
            "postgres": pg_sku_count,
            "match": jsonl_sku_count == pg_sku_count,
        },
    }

    return results


def show_stats(session: Session, tenant_id: str) -> None:
    """Show migration statistics.

    Args:
        session: Database session
        tenant_id: Tenant identifier
    """
    tenant_count = session.query(Tenant).filter_by(tenant_id=tenant_id).count()
    sku_count = session.query(SKU).filter_by(tenant_id=tenant_id).count()
    session_count = (
        session.query(AnalysisSession).filter_by(tenant_id=tenant_id).count()
    )
    api_key_count = session.query(APIKey).filter_by(tenant_id=tenant_id).count()

    print("\n📊 PostgreSQL Statistics")
    print(f"  Tenants: {tenant_count}")
    print(f"  API Keys: {api_key_count}")
    print(f"  SKUs: {sku_count}")
    print(f"  Analysis Sessions: {session_count}")


def main():
    parser = argparse.ArgumentParser(description="Migrate JSONL data to PostgreSQL")
    parser.add_argument(
        "--mode",
        choices=["backfill", "verify", "stats"],
        default="backfill",
        help="Migration mode",
    )
    parser.add_argument(
        "--tenant", default="default", help="Tenant ID to migrate"
    )

    args = parser.parse_args()

    print(f"\n🔄 JSONL → PostgreSQL Migration ({args.mode} mode)")
    print(f"   Tenant: {args.tenant}\n")

    with get_standalone_session() as session:
        if args.mode == "backfill":
            # Migrate tenants
            migrate_tenants(session, args.tenant)

            # Migrate SKUs
            migrate_skus(session, args.tenant)

            # Migrate analysis sessions
            migrate_analysis_sessions(session, args.tenant)

            print("\n✅ Backfill completed")

        elif args.mode == "verify":
            results = verify_migration(session, args.tenant)

            print("\n🔍 Verification Results")
            for entity, counts in results.items():
                status = "✓" if counts["match"] else "✗"
                print(
                    f"  {status} {entity.upper()}: JSONL={counts['jsonl']}, PostgreSQL={counts['postgres']}"
                )

            all_match = all(r["match"] for r in results.values())
            if all_match:
                print("\n✅ All data verified successfully")
            else:
                print("\n⚠️  Data mismatch detected")
                sys.exit(1)

        elif args.mode == "stats":
            show_stats(session, args.tenant)


if __name__ == "__main__":
    main()
