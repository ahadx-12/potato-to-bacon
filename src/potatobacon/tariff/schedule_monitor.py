"""HTS schedule change detection and SKU alerting.

Implements the nightly monitoring job that:
1. Fetches the latest USITC HTS edition
2. Diffs it against the previous version
3. Cross-references changed headings against tenant SKU classifications
4. Generates alerts for affected SKUs

This is the engine behind the $X/month/SKU recurring revenue:
importers subscribe to be notified when tariff changes affect their products.

Usage::

    monitor = ScheduleMonitor(store_dir=Path("data/hts_extract/usitc"))
    alerts = monitor.check_for_changes()
    # alerts = [{"tenant_id": "acme", "sku_id": "SKU-001", "change": {...}}, ...]
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from potatobacon.tariff.hts_ingest.versioned_store import (
    HTSEditionDiff,
    VersionedHTSStore,
)
from potatobacon.tariff.sku_store import get_default_sku_store

logger = logging.getLogger(__name__)


class ScheduleMonitor:
    """Monitors HTS schedule changes and alerts affected tenants."""

    def __init__(
        self,
        store_dir: Path | None = None,
    ) -> None:
        self.hts_store = VersionedHTSStore(store_dir)

    def check_for_changes(
        self,
        new_edition_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the schedule change detection pipeline.

        1. Get current and previous editions
        2. Compute diff
        3. Load all SKU classifications
        4. Find affected SKUs
        5. Return alerts

        Parameters
        ----------
        new_edition_id : str, optional
            Override the "new" edition to diff against.  If not given,
            uses the latest registered edition.
        """
        current = new_edition_id or self.hts_store.get_current_edition()
        previous = self.hts_store.get_previous_edition()

        if not current or not previous:
            logger.info("Need at least 2 editions to compute diff")
            return []

        if current == previous:
            logger.info("No new edition to diff")
            return []

        # Compute diff
        diff = self.hts_store.compute_diff(previous, current)
        if diff.total_changes == 0:
            logger.info("No changes between %s and %s", previous, current)
            return []

        logger.info(
            "Detected %d changes between %s → %s (%d added, %d removed, %d rate changes)",
            diff.total_changes,
            previous,
            current,
            len(diff.added),
            len(diff.removed),
            len(diff.rate_changed),
        )

        # Load SKU classifications
        sku_store = get_default_sku_store()
        all_skus = sku_store.list(limit=200)

        sku_classifications: Dict[str, str] = {}
        for sku in all_skus:
            if sku.current_hts:
                sku_classifications[sku.sku_id] = sku.current_hts

        if not sku_classifications:
            logger.info("No SKUs with HTS classifications found")
            return []

        # Find affected SKUs
        raw_alerts = self.hts_store.find_affected_skus(diff, sku_classifications)

        # Enrich alerts with metadata
        alerts: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()
        for raw in raw_alerts:
            sku = sku_store.get(raw["sku_id"])
            alerts.append({
                "sku_id": raw["sku_id"],
                "current_hts": raw["current_hts"],
                "affected_heading": raw["affected_heading"],
                "change_type": raw["change_type"],
                "change_detail": raw["change_detail"],
                "sku_description": sku.description if sku else None,
                "detected_at": now,
                "from_edition": diff.from_edition,
                "to_edition": diff.to_edition,
            })

        logger.info("Generated %d alerts for %d SKUs", len(alerts), len(sku_classifications))
        return alerts

    def get_diff_summary(
        self, from_edition: str, to_edition: str
    ) -> Dict[str, Any]:
        """Get a summary of changes between two editions."""
        diff = self.hts_store.compute_diff(from_edition, to_edition)
        return diff.to_dict()
