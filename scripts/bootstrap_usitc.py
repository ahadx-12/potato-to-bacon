"""Bootstrap USITC HTS data into the live tariff context.

Fetches the current USITC edition, parses it through the Sprint A
vocabulary bridge to generate guard tokens, and registers the result
in the VersionedHTSStore under context ID ``HTS_US_LIVE``.

Usage::

    # Standalone execution
    python -m scripts.bootstrap_usitc

    # Importable for cron scheduling
    from scripts.bootstrap_usitc import bootstrap_usitc
    result = bootstrap_usitc()
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from potatobacon.tariff.hts_ingest.usitc_context import (
    _tariff_line_to_policy_atom,
    load_usitc_context,
)
from potatobacon.tariff.hts_ingest.usitc_fetcher import USITCFetcher, USITCEdition
from potatobacon.tariff.hts_ingest.usitc_parser import parse_usitc_edition
from potatobacon.tariff.hts_ingest.versioned_store import VersionedHTSStore
from potatobacon.tariff.atom_utils import duty_rate_index

logger = logging.getLogger(__name__)

LIVE_CONTEXT_ID = "HTS_US_LIVE"
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[0].parent
    / "src"
    / "potatobacon"
    / "tariff"
    / "contexts"
    / "manifests"
)


def _write_context_manifest(
    edition: USITCEdition,
    atom_count: int,
    rate_line_count: int,
) -> Path:
    """Create or update the HTS_US_LIVE context manifest file.

    Returns the path to the written manifest.
    """
    manifest = {
        "context_id": LIVE_CONTEXT_ID,
        "domain": "tariff",
        "jurisdiction": "US",
        "effective_from": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "effective_to": None,
        "description": f"USITC HTS Live ({edition.edition_id}), {atom_count} atoms from {rate_line_count} rate lines",
        "loader": {
            "type": "python",
            "callable": "potatobacon.tariff.hts_ingest.usitc_context:load_usitc_context",
        },
        "sources": [edition.file_path],
        "revision_id": edition.edition_id,
        "edition_id": edition.edition_id,
        "edition_sha256": edition.sha256,
        "atom_count": atom_count,
        "rate_line_count": rate_line_count,
    }

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFESTS_DIR / f"{LIVE_CONTEXT_ID}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Wrote context manifest to %s", manifest_path)
    return manifest_path


def bootstrap_usitc(
    *,
    data_dir: Optional[Path] = None,
    store_dir: Optional[Path] = None,
    url: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Fetch, parse, and register USITC data as HTS_US_LIVE.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to store downloaded USITC data.
    store_dir : Path, optional
        Directory for the VersionedHTSStore.
    url : str, optional
        Override the USITC bulk data URL (useful for testing).
    max_retries : int
        Number of fetch retries on network failure.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    Dict with bootstrap results: edition_id, atom_count, rate_line_count, etc.
    """
    logger.info("Starting USITC bootstrap...")

    # Step 1: Fetch current edition
    fetcher = USITCFetcher(data_dir=data_dir)
    fetch_kwargs: Dict[str, Any] = {
        "max_retries": max_retries,
        "timeout": timeout,
    }
    if url is not None:
        fetch_kwargs["url"] = url

    try:
        edition = fetcher.fetch_current_edition(**fetch_kwargs)
    except ConnectionError as exc:
        logger.error("Failed to fetch USITC data: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error during USITC fetch: %s", exc)
        raise

    logger.info(
        "Fetched USITC edition %s: %d records",
        edition.edition_id,
        edition.record_count,
    )

    # Step 2: Load records and parse through the vocabulary bridge
    raw_records = fetcher.load_local_edition(edition.edition_id)
    lines, parse_errors = _parse_with_error_handling(raw_records)

    if parse_errors:
        logger.warning(
            "Encountered %d parse errors during USITC ingest (continuing with %d valid lines)",
            len(parse_errors),
            len(lines),
        )

    if not lines:
        logger.error("No valid tariff lines parsed from USITC edition %s", edition.edition_id)
        raise ValueError(f"Empty parse result for edition {edition.edition_id}")

    # Step 3: Register in VersionedHTSStore
    store = VersionedHTSStore(store_dir or data_dir)
    store.register_edition(
        edition.edition_id,
        source_url=edition.source_url,
        record_count=edition.record_count,
        sha256=edition.sha256,
        file_path=edition.file_path,
    )

    # Step 4: Convert to PolicyAtoms for counting
    atoms = [_tariff_line_to_policy_atom(line) for line in lines]
    duty_rates = duty_rate_index(atoms)

    # Step 5: Write context manifest
    _write_context_manifest(
        edition=edition,
        atom_count=len(atoms),
        rate_line_count=len(lines),
    )

    result = {
        "edition_id": edition.edition_id,
        "record_count": edition.record_count,
        "rate_line_count": len(lines),
        "atom_count": len(atoms),
        "duty_rate_count": len(duty_rates),
        "parse_errors": len(parse_errors),
        "sha256": edition.sha256,
        "context_id": LIVE_CONTEXT_ID,
    }

    logger.info(
        "USITC bootstrap complete: edition=%s, atoms=%d, rate_lines=%d, errors=%d",
        edition.edition_id,
        len(atoms),
        len(lines),
        len(parse_errors),
    )
    return result


def _parse_with_error_handling(
    records: List[Dict[str, Any]],
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Parse USITC records with per-line error handling.

    Returns (valid_lines, error_records) so we can log warnings
    without failing the entire ingest on individual line parse errors.
    """
    from potatobacon.tariff.hts_ingest.usitc_parser import (
        extract_chapter,
        extract_heading,
        usitc_record_to_tariff_line,
    )
    import re

    lines = []
    errors: List[Dict[str, Any]] = []
    hierarchy: Dict[int, str] = {}

    for i, record in enumerate(records):
        try:
            htsno = str(record.get("htsno") or "").strip()
            description = str(record.get("description") or "").strip()
            indent = int(record.get("indent") or 0)
            general = str(record.get("general") or "").strip()

            if not htsno or not description:
                continue

            # Update hierarchy tracker
            hierarchy[indent] = description
            for level in list(hierarchy.keys()):
                if level > indent:
                    del hierarchy[level]

            parent_descs = [
                hierarchy[j] for j in sorted(hierarchy.keys()) if j < indent
            ]

            # Skip non-rate lines
            digits = re.sub(r"\D", "", htsno)
            if len(digits) < 8 or not general:
                continue

            line = usitc_record_to_tariff_line(
                record,
                parent_descriptions=parent_descs,
            )
            if line is not None:
                lines.append(line)

        except Exception as exc:
            htsno_safe = str(record.get("htsno", f"record_{i}"))
            logger.warning(
                "Parse error on record %d (htsno=%s): %s",
                i, htsno_safe, exc,
            )
            errors.append({
                "index": i,
                "htsno": htsno_safe,
                "error": str(exc),
            })

    lines.sort(key=lambda l: (l.chapter, l.heading, l.hts_code, l.source_id))
    return lines, errors


def bootstrap_from_records(
    records: List[Dict[str, Any]],
    *,
    store_dir: Optional[Path] = None,
    edition_id: str = "USITC_FIXTURE",
) -> Dict[str, Any]:
    """Bootstrap from pre-loaded records (for testing without network).

    Parameters
    ----------
    records : list
        Raw USITC-format records.
    store_dir : Path, optional
        Directory for the VersionedHTSStore.
    edition_id : str
        Edition identifier to use.

    Returns
    -------
    Dict with bootstrap results.
    """
    import tempfile

    store_path = store_dir or Path(tempfile.mkdtemp(prefix="usitc_test_"))
    store = VersionedHTSStore(store_path)

    # Save records
    sha = store.save_edition_records(edition_id, records)

    # Register edition
    jsonl_path = store_path / f"{edition_id}.jsonl"
    store.register_edition(
        edition_id,
        source_url="fixture://test",
        record_count=len(records),
        sha256=sha,
        file_path=str(jsonl_path),
    )

    # Parse with error handling
    lines, parse_errors = _parse_with_error_handling(records)
    atoms = [_tariff_line_to_policy_atom(line) for line in lines]
    duty_rates = duty_rate_index(atoms)

    return {
        "edition_id": edition_id,
        "record_count": len(records),
        "rate_line_count": len(lines),
        "atom_count": len(atoms),
        "duty_rate_count": len(duty_rates),
        "parse_errors": len(parse_errors),
        "sha256": sha,
        "context_id": LIVE_CONTEXT_ID,
        "atoms": atoms,
        "duty_rates": duty_rates,
        "store_dir": store_path,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        result = bootstrap_usitc()
        print(f"\nBootstrap complete:")
        print(f"  Edition:    {result['edition_id']}")
        print(f"  Records:    {result['record_count']}")
        print(f"  Rate lines: {result['rate_line_count']}")
        print(f"  Atoms:      {result['atom_count']}")
        print(f"  Duty rates: {result['duty_rate_count']}")
        print(f"  Errors:     {result['parse_errors']}")
        print(f"  SHA-256:    {result['sha256'][:16]}...")
        print(f"  Context:    {result['context_id']}")
    except Exception as exc:
        logger.error("Bootstrap failed: %s", exc)
        sys.exit(1)
