"""MFN rate lookup store with hierarchical HTS code resolution.

Provides automatic base-rate and Special-column lookups for any HTS code,
backed by USITC tariff schedule data.  The store supports hierarchical
fallback: exact 10-digit → 8-digit → 6-digit → 4-digit heading → 2-digit
chapter.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from potatobacon.tariff.hts_ingest.usitc_parser import (
    ParsedDutyRate,
    duty_rate_as_float,
    parse_special_rates,
    parse_usitc_duty_rate,
)

logger = logging.getLogger(__name__)

_DIGITS_RE = re.compile(r"\D")


def _normalize(code: str) -> str:
    """Strip all non-digit characters from an HTS code."""
    return _DIGITS_RE.sub("", str(code))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RateEntry:
    """Single stored rate record."""

    hts_code: str
    hts_digits: str
    description: str
    general_rate: ParsedDutyRate
    special_rates: Dict[str, str]  # program_code → rate string


@dataclass(frozen=True)
class RateLookupResult:
    """Result of looking up an MFN (General) duty rate."""

    found: bool
    hts_code_matched: str
    match_level: str  # exact | 10-digit | 8-digit | 6-digit | 4-digit | chapter | not_found
    ad_valorem_rate: Optional[float]
    parsed_rate: Optional[ParsedDutyRate]
    is_specific_rate: bool
    is_compound_rate: bool
    requires_manual_review: bool
    description: str
    warning: Optional[str] = None


@dataclass(frozen=True)
class SpecialRateLookupResult:
    """Result of looking up a Special-column preferential rate."""

    found: bool
    program_code: str
    rate_string: str
    ad_valorem_rate: Optional[float]
    is_free: bool
    hts_code_matched: str


# ---------------------------------------------------------------------------
# Country → Special-column program code mapping
# ---------------------------------------------------------------------------
_COUNTRY_TO_SPECIAL: Dict[str, List[str]] = {
    "AU": ["AU"],
    "BH": ["BH"],
    "CA": ["CA"],
    "CL": ["CL"],
    "CO": ["CO"],
    "IL": ["IL"],
    "JO": ["JO"],
    "KR": ["KR"],
    "MA": ["MA"],
    "MX": ["MX"],
    "OM": ["OM"],
    "PA": ["PA"],
    "PE": ["PE"],
    "SG": ["S", "SG"],
    # GSP beneficiaries check "A" or "A+" codes
    "BD": ["A"], "KH": ["A"], "IN": ["A"], "ID": ["A"],
    "PH": ["A"], "TH": ["A"], "LK": ["A"], "PK": ["A"],
    "EG": ["A"], "TR": ["A"], "ZA": ["A"], "BR": ["A"],
    "AR": ["A"], "EC": ["A"], "BO": ["A"], "UY": ["A"],
}


def country_to_special_codes(country: str) -> List[str]:
    """Map an ISO-2 country code to USITC Special-column program codes."""
    return _COUNTRY_TO_SPECIAL.get(country.upper(), [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _not_found(hts_code: str) -> RateLookupResult:
    return RateLookupResult(
        found=False,
        hts_code_matched="",
        match_level="not_found",
        ad_valorem_rate=None,
        parsed_rate=None,
        is_specific_rate=False,
        is_compound_rate=False,
        requires_manual_review=False,
        description="",
        warning=f"HTS code {hts_code} not found in rate store",
    )


def _make_result(entry: RateEntry, level: str) -> RateLookupResult:
    rate = entry.general_rate
    av = duty_rate_as_float(rate)
    is_specific = rate.specific_amount is not None and rate.ad_valorem_pct is None
    is_compound = rate.is_compound
    needs_review = is_specific or is_compound or rate.is_unknown

    warning: Optional[str] = None
    if is_specific:
        warning = f"Specific rate: {rate.raw} — cannot auto-convert to ad valorem"
    elif is_compound:
        warning = f"Compound rate: {rate.raw} — using ad valorem component only"

    return RateLookupResult(
        found=True,
        hts_code_matched=entry.hts_code,
        match_level=level,
        ad_valorem_rate=av,
        parsed_rate=rate,
        is_specific_rate=is_specific,
        is_compound_rate=is_compound,
        requires_manual_review=needs_review,
        description=entry.description,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# MFNRateStore
# ---------------------------------------------------------------------------
class MFNRateStore:
    """Indexed store for MFN (General) and Special-column duty rates.

    Supports hierarchical lookup: queries are tried at progressively shorter
    digit prefixes until a match is found.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, RateEntry] = {}

    # -- loaders -----------------------------------------------------------

    def load_seed(self, path: Path) -> int:
        """Load rates from a JSON seed file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for rec in data.get("rates", []):
            hts = str(rec.get("hts_code", ""))
            digits = _normalize(hts)
            if not digits:
                continue
            general = parse_usitc_duty_rate(str(rec.get("general", "")))

            raw_special = rec.get("special_rates", rec.get("special", ""))
            if isinstance(raw_special, str):
                special = parse_special_rates(raw_special)
            elif isinstance(raw_special, dict):
                special = {str(k): str(v) for k, v in raw_special.items()}
            else:
                special = {}

            self._entries[digits] = RateEntry(
                hts_code=hts,
                hts_digits=digits,
                description=str(rec.get("description", "")),
                general_rate=general,
                special_rates=special,
            )
            count += 1
        logger.info("Loaded %d rate entries from seed %s", count, path.name)
        return count

    def load_full_seed(self, path: Path) -> int:
        """Load rates from the comprehensive full HTS rate seed JSON.

        This is the primary load path after running ``scripts/fetch_full_hts.py``.
        It handles the ``general_structured`` field for specific/compound rates.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for rec in data.get("rates", []):
            hts = str(rec.get("hts_code", ""))
            digits = _normalize(hts)
            if not digits:
                continue
            general = parse_usitc_duty_rate(str(rec.get("general", "")))

            raw_special = rec.get("special_rates", rec.get("special", ""))
            if isinstance(raw_special, str):
                special = parse_special_rates(raw_special)
            elif isinstance(raw_special, dict):
                special = {str(k): str(v) for k, v in raw_special.items()}
            else:
                special = {}

            self._entries[digits] = RateEntry(
                hts_code=hts,
                hts_digits=digits,
                description=str(rec.get("description", "")),
                general_rate=general,
                special_rates=special,
            )
            count += 1
        logger.info("Loaded %d rate entries from full seed %s", count, path.name)
        return count

    def load_usitc_edition(self, jsonl_path: Path) -> int:
        """Load rates directly from a raw USITC edition JSONL file.

        This parses the raw USITC records using the same parsing pipeline
        as fetch_full_hts.py but without needing the intermediate seed JSON.
        """
        import re as _re

        count = 0
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                htsno = str(record.get("htsno") or "").strip()
                if not htsno:
                    continue
                digits = _re.sub(r"\D", "", htsno)
                if len(digits) < 8:
                    continue
                indent = int(record.get("indent") or 0)
                general_str = str(record.get("general") or "").strip()
                if indent < 1 and not general_str:
                    continue

                general = parse_usitc_duty_rate(general_str)
                special_raw = str(record.get("special") or "").strip()
                special = parse_special_rates(special_raw)

                self._entries[digits] = RateEntry(
                    hts_code=htsno,
                    hts_digits=digits,
                    description=str(record.get("description") or "").strip(),
                    general_rate=general,
                    special_rates=special,
                )
                count += 1
        logger.info("Loaded %d rate entries from USITC edition %s", count, jsonl_path.name)
        return count

    def load_chapter_jsonl(self, path: Path) -> int:
        """Load rates from an internal-format chapter JSONL file."""
        count = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line.strip())
                hts = str(rec.get("hts_code", rec.get("htsno", "")))
                digits = _normalize(hts)
                if not digits or len(digits) < 4:
                    continue

                general_str = str(
                    rec.get("base_duty_rate", rec.get("general", ""))
                )
                raw_special = rec.get("special_rates", rec.get("special", ""))
                if isinstance(raw_special, str):
                    special = parse_special_rates(raw_special)
                elif isinstance(raw_special, dict):
                    special = {str(k): str(v) for k, v in raw_special.items()}
                else:
                    special = {}

                general = parse_usitc_duty_rate(general_str)
                self._entries[digits] = RateEntry(
                    hts_code=hts,
                    hts_digits=digits,
                    description=str(rec.get("description", "")),
                    general_rate=general,
                    special_rates=special,
                )
                count += 1
        logger.info("Loaded %d rate entries from %s", count, path.name)
        return count

    # -- lookups -----------------------------------------------------------

    def lookup(self, hts_code: str) -> RateLookupResult:
        """Look up the MFN (General) duty rate with hierarchical fallback."""
        digits = _normalize(hts_code)
        if not digits:
            return _not_found(hts_code)

        # Try progressively shorter prefixes
        _levels = [
            (len(digits), "exact"),
            (10, "10-digit"),
            (8, "8-digit"),
            (6, "6-digit"),
            (4, "4-digit"),
            (2, "chapter"),
        ]
        for length, level in _levels:
            if length > len(digits):
                continue
            prefix = digits[:length]
            if prefix in self._entries:
                return _make_result(self._entries[prefix], level)

        return _not_found(hts_code)

    def lookup_special(
        self, hts_code: str, country: str
    ) -> SpecialRateLookupResult:
        """Look up the Special-column rate for a country (or program code)."""
        digits = _normalize(hts_code)
        codes = country_to_special_codes(country.upper())
        if not codes:
            codes = [country.upper()]

        _levels = [len(digits), 10, 8, 6, 4, 2]
        for length in _levels:
            if length > len(digits):
                continue
            prefix = digits[:length]
            if prefix not in self._entries:
                continue
            entry = self._entries[prefix]
            for code in codes:
                if code in entry.special_rates:
                    rate_str = entry.special_rates[code]
                    parsed = parse_usitc_duty_rate(rate_str)
                    return SpecialRateLookupResult(
                        found=True,
                        program_code=code,
                        rate_string=rate_str,
                        ad_valorem_rate=duty_rate_as_float(parsed),
                        is_free=parsed.is_free,
                        hts_code_matched=entry.hts_code,
                    )
        return SpecialRateLookupResult(
            found=False,
            program_code=codes[0] if codes else country.upper(),
            rate_string="",
            ad_valorem_rate=None,
            is_free=False,
            hts_code_matched="",
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def iter_entries(self) -> List[RateEntry]:
        """Return all indexed entries as a deterministic list."""
        return [self._entries[key] for key in sorted(self._entries.keys())]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
def _default_seed_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "hts_extract" / "hts_rates_seed.json"


def _full_seed_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "hts_extract" / "hts_rates_full.json"


def _usitc_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "hts_extract" / "usitc"


def _chapter_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "hts_extract" / "full_chapters"


@lru_cache(maxsize=1)
def get_rate_store() -> MFNRateStore:
    """Return a cached, fully-loaded :class:`MFNRateStore`.

    Loading priority (each layer overrides the previous):
    1. Full USITC seed (``hts_rates_full.json``) — output of fetch_full_hts.py
    2. Raw USITC edition JSONL files from ``data/hts_extract/usitc/``
    3. Chapter JSONL files from ``data/hts_extract/full_chapters/``
    4. Original 136-entry seed (``hts_rates_seed.json``) — offline fallback
    """
    store = MFNRateStore()
    loaded_full = False

    # 1. Try full USITC seed first (best source — complete 10,000+ entries)
    full_seed = _full_seed_path()
    if full_seed.exists():
        try:
            count = store.load_full_seed(full_seed)
            if count > 1000:
                loaded_full = True
                logger.info("Full USITC seed loaded: %d entries", count)
        except Exception:
            logger.warning("Failed to load full seed %s", full_seed.name, exc_info=True)

    # 2. Try raw USITC edition JSONL (if no full seed available)
    if not loaded_full:
        usitc_dir = _usitc_dir()
        if usitc_dir.is_dir():
            # Find latest edition (not _raw.json, not _meta.json)
            jsonl_files = sorted(
                [p for p in usitc_dir.glob("USITC_*.jsonl") if "_raw" not in p.name],
                reverse=True,
            )
            for jsonl_path in jsonl_files[:1]:  # only load latest
                try:
                    count = store.load_usitc_edition(jsonl_path)
                    if count > 1000:
                        loaded_full = True
                        logger.info("USITC edition loaded: %d entries from %s", count, jsonl_path.name)
                except Exception:
                    logger.warning("Failed to load USITC edition %s", jsonl_path.name, exc_info=True)

    # 3. Load chapter files (broader coverage, or supplements full seed)
    ch_dir = _chapter_dir()
    if ch_dir.is_dir():
        for p in sorted(ch_dir.glob("ch*.jsonl")):
            try:
                store.load_chapter_jsonl(p)
            except Exception:
                logger.warning("Failed to load chapter file %s", p.name, exc_info=True)

    # 4. Load original seed (overrides where both exist — curated data)
    seed = _default_seed_path()
    if seed.exists():
        try:
            store.load_seed(seed)
        except Exception:
            logger.warning("Failed to load seed file %s", seed.name, exc_info=True)

    logger.info("Rate store ready: %d entries", store.entry_count)
    return store
