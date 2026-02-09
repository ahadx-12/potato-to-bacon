"""Versioned HTS data store with change tracking.

Tracks HTS schedule versions over time and computes diffs between
editions to detect which tariff lines changed, were added, or removed.
This powers the schedule monitoring product — when a new USITC revision
drops, we diff it against the previous version and flag affected SKUs.

Storage layout::

    data/hts_extract/usitc/
        editions.json                  # Edition registry
        USITC_20250101_000000.jsonl    # Raw records per edition
        USITC_20250101_000000_meta.json
        diffs/
            diff_USITC_20250101_to_20250215.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[3].parent / "data" / "hts_extract" / "usitc"


@dataclass
class HTSLineChange:
    """A single change between two HTS editions."""

    hts_code: str
    change_type: str  # "added" | "removed" | "rate_changed" | "description_changed"
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    old_rate: Optional[str] = None
    new_rate: Optional[str] = None


@dataclass
class HTSEditionDiff:
    """Diff between two HTS editions."""

    from_edition: str
    to_edition: str
    computed_at: str
    total_lines_before: int
    total_lines_after: int
    added: List[HTSLineChange] = field(default_factory=list)
    removed: List[HTSLineChange] = field(default_factory=list)
    rate_changed: List[HTSLineChange] = field(default_factory=list)
    description_changed: List[HTSLineChange] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.removed) + len(self.rate_changed) + len(self.description_changed)

    def affected_chapters(self) -> Set[str]:
        """Return set of chapter numbers that had changes."""
        chapters: Set[str] = set()
        for change in self.added + self.removed + self.rate_changed + self.description_changed:
            digits = change.hts_code.replace(".", "").replace(" ", "")
            if len(digits) >= 2:
                chapters.add(digits[:2])
        return chapters

    def affected_headings(self) -> Set[str]:
        """Return set of 4-digit headings that had changes."""
        headings: Set[str] = set()
        for change in self.added + self.removed + self.rate_changed + self.description_changed:
            digits = change.hts_code.replace(".", "").replace(" ", "")
            if len(digits) >= 4:
                headings.add(digits[:4])
        return headings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_edition": self.from_edition,
            "to_edition": self.to_edition,
            "computed_at": self.computed_at,
            "total_lines_before": self.total_lines_before,
            "total_lines_after": self.total_lines_after,
            "summary": {
                "added": len(self.added),
                "removed": len(self.removed),
                "rate_changed": len(self.rate_changed),
                "description_changed": len(self.description_changed),
                "total_changes": self.total_changes,
                "affected_chapters": sorted(self.affected_chapters()),
            },
            "changes": {
                "added": [
                    {"hts_code": c.hts_code, "new_value": c.new_value, "new_rate": c.new_rate}
                    for c in self.added
                ],
                "removed": [
                    {"hts_code": c.hts_code, "old_value": c.old_value, "old_rate": c.old_rate}
                    for c in self.removed
                ],
                "rate_changed": [
                    {"hts_code": c.hts_code, "old_rate": c.old_rate, "new_rate": c.new_rate}
                    for c in self.rate_changed
                ],
                "description_changed": [
                    {"hts_code": c.hts_code, "old_value": c.old_value, "new_value": c.new_value}
                    for c in self.description_changed
                ],
            },
        }


class VersionedHTSStore:
    """Manages versioned HTS data with diff tracking."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or DEFAULT_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_dir = self.store_dir / "diffs"
        self.diffs_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.store_dir / "editions.json"
        self._registry = self._load_registry()

    # -- Edition management ------------------------------------------------

    def register_edition(
        self,
        edition_id: str,
        *,
        source_url: str = "",
        record_count: int = 0,
        sha256: str = "",
        file_path: str = "",
    ) -> None:
        """Register a new edition in the store."""
        self._registry["editions"][edition_id] = {
            "edition_id": edition_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "source_url": source_url,
            "record_count": record_count,
            "sha256": sha256,
            "file_path": file_path,
        }
        self._registry["current"] = edition_id
        self._save_registry()

    def get_current_edition(self) -> Optional[str]:
        """Return the current (latest) edition ID."""
        return self._registry.get("current")

    def get_previous_edition(self) -> Optional[str]:
        """Return the edition before the current one."""
        editions = sorted(self._registry.get("editions", {}).keys())
        current = self._registry.get("current")
        if not current or current not in editions:
            return None
        idx = editions.index(current)
        return editions[idx - 1] if idx > 0 else None

    def list_editions(self) -> List[Dict[str, Any]]:
        """Return all registered editions sorted by ID."""
        editions = list(self._registry.get("editions", {}).values())
        editions.sort(key=lambda e: e.get("edition_id", ""))
        return editions

    # -- Data loading ------------------------------------------------------

    def load_edition_records(self, edition_id: str) -> Dict[str, Dict[str, Any]]:
        """Load an edition's records as {hts_code: record} dict."""
        jsonl_path = self.store_dir / f"{edition_id}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Edition data not found: {jsonl_path}")

        records: Dict[str, Dict[str, Any]] = {}
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                hts_code = str(record.get("htsno") or record.get("hts_code") or "")
                if hts_code:
                    records[hts_code] = record
        return records

    def save_edition_records(
        self,
        edition_id: str,
        records: List[Dict[str, Any]],
    ) -> str:
        """Save records for an edition and return its SHA-256 hash."""
        jsonl_path = self.store_dir / f"{edition_id}.jsonl"
        content_parts = []
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in records:
                line = json.dumps(record, sort_keys=True, separators=(",", ":"))
                f.write(line + "\n")
                content_parts.append(line)

        sha = hashlib.sha256("\n".join(content_parts).encode("utf-8")).hexdigest()
        return sha

    # -- Diff engine -------------------------------------------------------

    def compute_diff(
        self, from_edition: str, to_edition: str
    ) -> HTSEditionDiff:
        """Compute the diff between two HTS editions."""
        old_records = self.load_edition_records(from_edition)
        new_records = self.load_edition_records(to_edition)

        diff = HTSEditionDiff(
            from_edition=from_edition,
            to_edition=to_edition,
            computed_at=datetime.now(timezone.utc).isoformat(),
            total_lines_before=len(old_records),
            total_lines_after=len(new_records),
        )

        old_codes = set(old_records.keys())
        new_codes = set(new_records.keys())

        # Added
        for code in sorted(new_codes - old_codes):
            rec = new_records[code]
            diff.added.append(HTSLineChange(
                hts_code=code,
                change_type="added",
                new_value=str(rec.get("description") or ""),
                new_rate=str(rec.get("general") or rec.get("duty_rate") or ""),
            ))

        # Removed
        for code in sorted(old_codes - new_codes):
            rec = old_records[code]
            diff.removed.append(HTSLineChange(
                hts_code=code,
                change_type="removed",
                old_value=str(rec.get("description") or ""),
                old_rate=str(rec.get("general") or rec.get("duty_rate") or ""),
            ))

        # Changed
        for code in sorted(old_codes & new_codes):
            old = old_records[code]
            new = new_records[code]

            old_rate = str(old.get("general") or old.get("duty_rate") or "")
            new_rate = str(new.get("general") or new.get("duty_rate") or "")
            old_desc = str(old.get("description") or "")
            new_desc = str(new.get("description") or "")

            if old_rate != new_rate:
                diff.rate_changed.append(HTSLineChange(
                    hts_code=code,
                    change_type="rate_changed",
                    old_rate=old_rate,
                    new_rate=new_rate,
                ))
            if old_desc != new_desc:
                diff.description_changed.append(HTSLineChange(
                    hts_code=code,
                    change_type="description_changed",
                    old_value=old_desc,
                    new_value=new_desc,
                ))

        # Save diff
        diff_filename = f"diff_{from_edition}_to_{to_edition}.json"
        diff_path = self.diffs_dir / diff_filename
        diff_path.write_text(
            json.dumps(diff.to_dict(), indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Computed diff %s → %s: %d changes (%d added, %d removed, %d rate changes)",
            from_edition, to_edition, diff.total_changes,
            len(diff.added), len(diff.removed), len(diff.rate_changed),
        )
        return diff

    def find_affected_skus(
        self,
        diff: HTSEditionDiff,
        sku_classifications: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Given a diff and SKU→HTS code mappings, find affected SKUs.

        Parameters
        ----------
        diff : HTSEditionDiff
            The edition diff to check.
        sku_classifications : Dict[str, str]
            Maps SKU IDs to their current HTS classification codes.

        Returns
        -------
        List of affected SKU alerts.
        """
        affected_headings = diff.affected_headings()
        alerts: List[Dict[str, Any]] = []

        for sku_id, hts_code in sku_classifications.items():
            digits = hts_code.replace(".", "").replace(" ", "")
            heading = digits[:4] if len(digits) >= 4 else digits[:2]

            if heading in affected_headings:
                # Find the specific change
                for change_list in [diff.rate_changed, diff.added, diff.removed, diff.description_changed]:
                    for change in change_list:
                        change_heading = change.hts_code.replace(".", "").replace(" ", "")[:4]
                        if change_heading == heading:
                            alerts.append({
                                "sku_id": sku_id,
                                "current_hts": hts_code,
                                "affected_heading": heading,
                                "change_type": change.change_type,
                                "change_detail": {
                                    "hts_code": change.hts_code,
                                    "old_rate": change.old_rate,
                                    "new_rate": change.new_rate,
                                    "old_desc": change.old_value,
                                    "new_desc": change.new_value,
                                },
                            })
                            break
                    else:
                        continue
                    break

        return alerts

    # -- Internals ---------------------------------------------------------

    def _load_registry(self) -> Dict[str, Any]:
        if self._registry_path.exists():
            with self._registry_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {"editions": {}, "current": None}

    def _save_registry(self) -> None:
        with self._registry_path.open("w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, sort_keys=True)
