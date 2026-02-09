"""USITC HTS data fetcher.

Downloads the machine-readable Harmonized Tariff Schedule from the
USITC website (hts.usitc.gov) and stores it locally.  Supports both
the bulk JSON download and the REST search API.

Usage::

    fetcher = USITCFetcher(data_dir=Path("data/hts_extract/usitc"))
    edition = fetcher.fetch_current_edition()  # downloads full JSON
    records = fetcher.search("copper cathodes")  # REST search
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

USITC_BULK_URL = "https://hts.usitc.gov/reststop/getFullData"
USITC_SEARCH_URL = "https://hts.usitc.gov/reststop/search"
USITC_EXPORT_URL = "https://hts.usitc.gov/reststop/exportList"

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3].parent / "data" / "hts_extract" / "usitc"


@dataclass
class USITCEdition:
    """Metadata for a downloaded USITC HTS edition."""

    edition_id: str
    download_date: str
    source_url: str
    record_count: int
    sha256: str
    file_path: str


@dataclass
class USITCRecord:
    """A single USITC HTS record (raw from their API)."""

    htsno: str
    description: str
    indent: int
    general: str
    special: str
    other: str
    units: List[str] = field(default_factory=list)
    footnotes: List[Dict[str, Any]] = field(default_factory=list)
    statistical_suffix: str = ""

    @classmethod
    def from_api(cls, raw: Dict[str, Any]) -> "USITCRecord":
        """Parse a record from the USITC REST API response."""
        return cls(
            htsno=str(raw.get("htsno") or "").strip(),
            description=str(raw.get("description") or "").strip(),
            indent=int(raw.get("indent") or 0),
            general=str(raw.get("general") or "").strip(),
            special=str(raw.get("special") or "").strip(),
            other=str(raw.get("other") or "").strip(),
            units=list(raw.get("units") or []),
            footnotes=list(raw.get("footnotes") or []),
            statistical_suffix=str(raw.get("statisticalSuffix") or "").strip(),
        )


class USITCFetcher:
    """Fetches HTS data from the USITC website."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_current_edition(
        self,
        url: str = USITC_BULK_URL,
        *,
        max_retries: int = 3,
        timeout: int = 120,
    ) -> USITCEdition:
        """Download the full USITC HTS dataset.

        Returns an USITCEdition with metadata about the download.
        """
        now = datetime.now(timezone.utc)
        edition_id = f"USITC_{now.strftime('%Y%m%d_%H%M%S')}"

        raw_data = self._fetch_with_retry(url, max_retries=max_retries, timeout=timeout)
        records = json.loads(raw_data)

        if isinstance(records, dict) and "results" in records:
            records = records["results"]
        if not isinstance(records, list):
            raise ValueError(f"Unexpected USITC response format: {type(records)}")

        sha = hashlib.sha256(raw_data.encode("utf-8")).hexdigest()

        # Save raw JSON
        raw_path = self.data_dir / f"{edition_id}_raw.json"
        raw_path.write_text(raw_data, encoding="utf-8")

        # Save parsed JSONL
        jsonl_path = self.data_dir / f"{edition_id}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

        # Save edition metadata
        edition = USITCEdition(
            edition_id=edition_id,
            download_date=now.isoformat(),
            source_url=url,
            record_count=len(records),
            sha256=sha,
            file_path=str(jsonl_path),
        )

        meta_path = self.data_dir / f"{edition_id}_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "edition_id": edition.edition_id,
                    "download_date": edition.download_date,
                    "source_url": edition.source_url,
                    "record_count": edition.record_count,
                    "sha256": edition.sha256,
                    "file_path": edition.file_path,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Downloaded USITC edition %s: %d records, sha256=%s",
            edition_id,
            len(records),
            sha[:16],
        )
        return edition

    def search(
        self, keyword: str, *, timeout: int = 30
    ) -> List[USITCRecord]:
        """Search the USITC REST API for HTS records matching a keyword."""
        from urllib.parse import quote

        url = f"{USITC_SEARCH_URL}?keyword={quote(keyword)}"
        raw = self._fetch_with_retry(url, max_retries=2, timeout=timeout)
        data = json.loads(raw)

        results = data if isinstance(data, list) else data.get("results", [])
        return [USITCRecord.from_api(r) for r in results]

    def load_local_edition(self, edition_id: str) -> List[Dict[str, Any]]:
        """Load a previously downloaded edition from local storage."""
        jsonl_path = self.data_dir / f"{edition_id}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Edition not found: {jsonl_path}")

        records = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def list_editions(self) -> List[Dict[str, Any]]:
        """List all locally stored USITC editions."""
        editions = []
        for meta_path in sorted(self.data_dir.glob("*_meta.json")):
            with meta_path.open("r", encoding="utf-8") as f:
                editions.append(json.load(f))
        return editions

    def _fetch_with_retry(
        self, url: str, *, max_retries: int = 3, timeout: int = 60
    ) -> str:
        """Fetch URL content with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                req = Request(url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8")
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "USITC fetch attempt %d failed: %s (retrying in %ds)",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise ConnectionError(
            f"Failed to fetch {url} after {max_retries} attempts"
        ) from last_error
