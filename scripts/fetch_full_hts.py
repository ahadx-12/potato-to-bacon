#!/usr/bin/env python3
"""Fetch the complete USITC HTS dataset and build a full rate seed.

Usage:
    python scripts/fetch_full_hts.py
    python scripts/fetch_full_hts.py --output data/hts_extract/hts_rates_full.json
    python scripts/fetch_full_hts.py --offline USITC_20250215_120000

Downloads the machine-readable HTS from hts.usitc.gov, parses every
8-digit tariff line, and writes a comprehensive rate seed JSON that the
MFNRateStore can load directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from potatobacon.tariff.hts_ingest.usitc_fetcher import USITCFetcher
from potatobacon.tariff.hts_ingest.usitc_parser import (
    ParsedDutyRate,
    duty_rate_as_float,
    extract_chapter,
    parse_special_rates,
    parse_usitc_duty_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fetch_full_hts")


def _rate_to_serializable(parsed: ParsedDutyRate) -> dict | str:
    """Convert a ParsedDutyRate to the seed JSON format."""
    if parsed.is_free:
        return "Free"
    if parsed.is_compound:
        result: dict = {"type": "compound"}
        if parsed.ad_valorem_pct is not None:
            result["ad_valorem"] = round(parsed.ad_valorem_pct / 100.0, 6)
        if parsed.specific_amount is not None:
            result["specific"] = round(parsed.specific_amount, 6)
            if parsed.specific_unit:
                result["per"] = parsed.specific_unit
        return result
    if parsed.specific_amount is not None and parsed.ad_valorem_pct is None:
        return {
            "type": "specific",
            "amount": round(parsed.specific_amount, 6),
            "per": parsed.specific_unit or "unit",
        }
    if parsed.ad_valorem_pct is not None:
        return f"{parsed.ad_valorem_pct}%"
    return parsed.raw


def _is_dutiable_line(record: dict) -> bool:
    """Check if a USITC record is an 8-digit tariff line with a rate."""
    htsno = str(record.get("htsno") or "").strip()
    digits = re.sub(r"\D", "", htsno)
    general = str(record.get("general") or "").strip()
    indent = int(record.get("indent") or 0)
    return len(digits) >= 8 and indent >= 1 and bool(general or htsno)


def build_seed_from_records(records: list[dict]) -> dict:
    """Transform raw USITC records into the rate seed format."""
    rates = []
    chapter_counts: dict[str, int] = defaultdict(int)
    rate_type_counts = {"ad_valorem": 0, "specific": 0, "compound": 0, "free": 0, "unknown": 0}

    for record in records:
        htsno = str(record.get("htsno") or "").strip()
        if not htsno:
            continue

        digits = re.sub(r"\D", "", htsno)
        general_str = str(record.get("general") or "").strip()
        description = str(record.get("description") or "").strip()
        indent = int(record.get("indent") or 0)

        # Only process 8+ digit rate lines
        if len(digits) < 8:
            continue
        if indent < 1 and not general_str:
            continue

        # Parse general rate
        parsed = parse_usitc_duty_rate(general_str)

        # Track rate types
        if parsed.is_free:
            rate_type_counts["free"] += 1
        elif parsed.is_compound:
            rate_type_counts["compound"] += 1
        elif parsed.specific_amount is not None and parsed.ad_valorem_pct is None:
            rate_type_counts["specific"] += 1
        elif parsed.ad_valorem_pct is not None:
            rate_type_counts["ad_valorem"] += 1
        elif parsed.is_unknown:
            rate_type_counts["unknown"] += 1

        # Parse special rates
        special_raw = str(record.get("special") or "").strip()
        special = parse_special_rates(special_raw)

        # Build rate entry
        entry: dict = {
            "hts_code": htsno,
            "general": general_str,
            "description": description,
        }

        # Add structured general rate for non-ad-valorem types
        general_structured = _rate_to_serializable(parsed)
        if isinstance(general_structured, dict):
            entry["general_structured"] = general_structured

        if special:
            entry["special_rates"] = special

        # Track unit of quantity
        units = record.get("units")
        if units:
            if isinstance(units, list) and units:
                entry["units"] = units
            elif isinstance(units, str) and units.strip():
                entry["units"] = [units.strip()]

        rates.append(entry)
        chapter = digits[:2]
        chapter_counts[chapter] += 1

    return {
        "metadata": {
            "source": "USITC Harmonized Tariff Schedule (full)",
            "generated_by": "scripts/fetch_full_hts.py",
            "description": f"Complete HTS rate data: {len(rates)} tariff lines across {len(chapter_counts)} chapters",
            "coverage": f"{len(rates)} entries",
            "rate_type_breakdown": rate_type_counts,
            "chapter_counts": dict(sorted(chapter_counts.items())),
        },
        "rates": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch full USITC HTS data and build comprehensive rate seed"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the rate seed JSON (default: data/hts_extract/hts_rates_full.json)",
    )
    parser.add_argument(
        "--offline",
        default=None,
        help="Use a previously downloaded USITC edition ID instead of fetching live",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for USITC raw downloads (default: data/hts_extract/usitc)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data" / "hts_extract" / "usitc"
    output_path = Path(args.output) if args.output else repo_root / "data" / "hts_extract" / "hts_rates_full.json"

    fetcher = USITCFetcher(data_dir=data_dir)

    if args.offline:
        logger.info("Loading offline edition: %s", args.offline)
        records = fetcher.load_local_edition(args.offline)
    else:
        logger.info("Fetching full USITC HTS dataset from hts.usitc.gov ...")
        edition = fetcher.fetch_current_edition()
        logger.info(
            "Downloaded edition %s: %d raw records",
            edition.edition_id,
            edition.record_count,
        )
        records = fetcher.load_local_edition(edition.edition_id)

    logger.info("Parsing %d raw USITC records ...", len(records))
    seed = build_seed_from_records(records)

    meta = seed["metadata"]
    logger.info("=" * 60)
    logger.info("USITC Full HTS Ingest Summary")
    logger.info("=" * 60)
    logger.info("  Total tariff lines:  %s", meta["coverage"])
    logger.info("  Rate type breakdown:")
    for rtype, count in meta["rate_type_breakdown"].items():
        logger.info("    %-12s  %d", rtype, count)
    logger.info("  Chapter coverage:")
    for ch, count in sorted(meta["chapter_counts"].items()):
        logger.info("    Chapter %s: %d entries", ch, count)
    logger.info("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(seed, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Wrote rate seed to %s", output_path)


if __name__ == "__main__":
    main()
