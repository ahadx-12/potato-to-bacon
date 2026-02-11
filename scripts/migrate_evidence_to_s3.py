#!/usr/bin/env python3
"""Migrate evidence blobs from local filesystem to S3.

Uploads evidence files to S3 and updates PostgreSQL evidence_metadata table
with S3 references while preserving SHA-256 content addressing.

Usage:
    # Migrate all evidence for a tenant
    python scripts/migrate_evidence_to_s3.py --tenant=default

    # Verify migration integrity
    python scripts/migrate_evidence_to_s3.py --tenant=default --verify

    # Dry run (no actual uploads)
    python scripts/migrate_evidence_to_s3.py --tenant=default --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

from sqlalchemy.orm import Session

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from potatobacon.db.models import EvidenceMetadata
from potatobacon.db.session import get_standalone_session
from potatobacon.storage.s3_backend import get_s3_backend
from potatobacon.tariff.evidence_store import get_default_evidence_store


def migrate_evidence(
    session: Session, tenant_id: str, dry_run: bool = False
) -> Dict[str, int]:
    """Migrate evidence blobs from local filesystem to S3.

    Args:
        session: Database session
        tenant_id: Tenant identifier
        dry_run: If True, don't actually upload to S3

    Returns:
        Dict with migration statistics
    """
    evidence_store = get_default_evidence_store()
    s3_backend = get_s3_backend()

    stats = {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0}

    # Iterate through all evidence records
    for evidence_id, record in evidence_store._records.items():
        stats["total"] += 1

        # Check if already migrated to PostgreSQL
        existing = (
            session.query(EvidenceMetadata).filter_by(evidence_id=evidence_id).first()
        )
        if existing:
            print(f"  ⏭️  Skipped {evidence_id[:12]}... (already in PostgreSQL)")
            stats["skipped"] += 1
            continue

        # Get local file path
        local_path = evidence_store.data_dir / evidence_id
        if not local_path.exists():
            print(f"  ⚠️  Missing file for {evidence_id[:12]}...")
            stats["failed"] += 1
            continue

        # Read content and verify SHA-256
        content = local_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()

        if actual_hash != evidence_id:
            print(
                f"  ❌ Hash mismatch for {evidence_id[:12]}... (expected {evidence_id[:12]}, got {actual_hash[:12]})"
            )
            stats["failed"] += 1
            continue

        if not dry_run:
            try:
                # Upload to S3
                s3_key = s3_backend.save_evidence(
                    tenant_id=tenant_id,
                    evidence_id=evidence_id,
                    content=content,
                    content_type=record.content_type,
                    original_filename=record.original_filename,
                )

                # Create PostgreSQL metadata record
                metadata = EvidenceMetadata(
                    evidence_id=evidence_id,
                    tenant_id=tenant_id,
                    original_filename=record.original_filename,
                    content_type=record.content_type,
                    byte_length=record.byte_length,
                    s3_bucket=s3_backend.bucket,
                    s3_key=s3_key,
                    uploaded_at=record.uploaded_at,
                    evidence_kind=record.evidence_kind,
                )
                session.add(metadata)
                session.commit()

                print(f"  ✓ Uploaded {evidence_id[:12]}... ({record.byte_length} bytes)")
                stats["uploaded"] += 1

            except Exception as exc:
                print(f"  ❌ Failed to upload {evidence_id[:12]}...: {exc}")
                stats["failed"] += 1
        else:
            print(
                f"  [DRY RUN] Would upload {evidence_id[:12]}... ({record.byte_length} bytes)"
            )
            stats["uploaded"] += 1

        # Commit every 10 records
        if stats["uploaded"] % 10 == 0 and not dry_run:
            session.commit()

    return stats


def verify_migration(session: Session, tenant_id: str) -> Dict[str, Any]:
    """Verify migration by comparing local files with S3.

    Args:
        session: Database session
        tenant_id: Tenant identifier

    Returns:
        Dict with verification results
    """
    evidence_store = get_default_evidence_store()
    s3_backend = get_s3_backend()

    local_count = len(evidence_store._records)
    pg_count = session.query(EvidenceMetadata).filter_by(tenant_id=tenant_id).count()

    verified = 0
    mismatches = []

    # Verify each record in PostgreSQL
    for metadata in session.query(EvidenceMetadata).filter_by(tenant_id=tenant_id):
        # Check local file
        local_path = evidence_store.data_dir / metadata.evidence_id
        if not local_path.exists():
            mismatches.append(
                f"{metadata.evidence_id[:12]}... - missing local file"
            )
            continue

        # Check S3 object
        s3_meta = s3_backend.head_object(metadata.s3_key)
        if not s3_meta:
            mismatches.append(
                f"{metadata.evidence_id[:12]}... - missing S3 object"
            )
            continue

        # Verify size matches
        local_size = local_path.stat().st_size
        s3_size = s3_meta["content_length"]

        if local_size != s3_size:
            mismatches.append(
                f"{metadata.evidence_id[:12]}... - size mismatch (local={local_size}, S3={s3_size})"
            )
            continue

        verified += 1

    return {
        "local_count": local_count,
        "postgres_count": pg_count,
        "verified": verified,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate evidence blobs to S3"
    )
    parser.add_argument(
        "--tenant", default="default", help="Tenant ID to migrate"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration integrity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (no actual uploads)",
    )

    args = parser.parse_args()

    print(f"\n📦 Evidence → S3 Migration")
    print(f"   Tenant: {args.tenant}")
    if args.dry_run:
        print("   Mode: DRY RUN\n")
    elif args.verify:
        print("   Mode: VERIFY\n")
    else:
        print("   Mode: MIGRATE\n")

    with get_standalone_session() as session:
        if args.verify:
            results = verify_migration(session, args.tenant)

            print(f"\n🔍 Verification Results")
            print(f"  Local files: {results['local_count']}")
            print(f"  PostgreSQL records: {results['postgres_count']}")
            print(f"  Verified: {results['verified']}")

            if results["mismatches"]:
                print(f"\n⚠️  Found {len(results['mismatches'])} mismatches:")
                for mismatch in results["mismatches"][:10]:
                    print(f"    • {mismatch}")
                if len(results["mismatches"]) > 10:
                    print(
                        f"    ... and {len(results['mismatches']) - 10} more"
                    )
                sys.exit(1)
            else:
                print("\n✅ All evidence verified successfully")

        else:
            stats = migrate_evidence(session, args.tenant, dry_run=args.dry_run)

            print(f"\n📊 Migration Statistics")
            print(f"  Total: {stats['total']}")
            print(f"  Uploaded: {stats['uploaded']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Failed: {stats['failed']}")

            if stats["failed"] > 0:
                print(f"\n⚠️  {stats['failed']} evidence blob(s) failed to migrate")
                sys.exit(1)
            else:
                print("\n✅ Migration completed successfully")


if __name__ == "__main__":
    main()
