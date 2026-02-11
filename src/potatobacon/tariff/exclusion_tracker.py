"""Presidential proclamation exclusion tracker.

Tracks Section 232 and Section 301 exclusions, checking whether a given
HTS code + origin country qualifies for a duty reduction or elimination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Exclusion:
    """Single presidential proclamation exclusion."""

    exclusion_id: str
    overlay_type: str  # "section_232" or "section_301"
    hts_codes: tuple[str, ...]
    product_description: str
    origin_countries: tuple[str, ...]  # empty = applies to all origins
    exclusion_rate_pct: float
    effective_date: str
    expiry_date: str
    status: str  # "active" or "expired"
    proclamation_number: str
    federal_register_citation: str
    requestor: str


@dataclass(frozen=True)
class ExclusionLookupResult:
    """Result of checking exclusions for a specific HTS code + origin."""

    active_exclusions: tuple[Exclusion, ...]
    expired_exclusions: tuple[Exclusion, ...]
    total_exclusion_relief_pct: float
    has_active_exclusion: bool
    has_expired_exclusion: bool


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "overlays" / "exclusions.json"


def _normalize_hts(code: str) -> str:
    """Strip dots/spaces, return digits only."""
    return "".join(ch for ch in str(code) if ch.isdigit())


def _normalize_country(code: str) -> str:
    return code.strip().upper()


def _hts_matches(hts_digits: str, exclusion_code: str) -> bool:
    """Check if an HTS code matches an exclusion code (prefix match)."""
    excl_digits = _normalize_hts(exclusion_code)
    if not excl_digits or not hts_digits:
        return False
    return hts_digits.startswith(excl_digits) or excl_digits.startswith(hts_digits)


def _parse_date(date_str: str) -> date | None:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _is_active(exclusion: Exclusion, reference_date: date | None = None) -> bool:
    """Check if an exclusion is currently active based on dates and status."""
    if exclusion.status == "expired":
        return False
    if exclusion.status != "active":
        return False
    if reference_date is None:
        reference_date = date.today()
    eff = _parse_date(exclusion.effective_date)
    exp = _parse_date(exclusion.expiry_date)
    if eff and reference_date < eff:
        return False
    if exp and reference_date > exp:
        return False
    return True


class ExclusionTracker:
    """Tracks presidential proclamation exclusions from Section 232/301 duties."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        path = Path(data_path) if data_path else _default_data_path()
        self._exclusions = _load_exclusions(path)

    @property
    def exclusions(self) -> tuple[Exclusion, ...]:
        return self._exclusions

    def check(
        self,
        hts_code: str,
        origin_country: str | None = None,
        *,
        reference_date: date | None = None,
    ) -> ExclusionLookupResult:
        """Check for applicable exclusions on a given HTS code and origin country."""
        hts_digits = _normalize_hts(hts_code)
        origin_norm = _normalize_country(origin_country) if origin_country else None

        active: list[Exclusion] = []
        expired: list[Exclusion] = []

        for excl in self._exclusions:
            # Check HTS match
            if not any(_hts_matches(hts_digits, code) for code in excl.hts_codes):
                continue

            # Check origin country (empty = all origins)
            if excl.origin_countries and origin_norm:
                if origin_norm not in excl.origin_countries:
                    continue
            elif excl.origin_countries and not origin_norm:
                # Exclusion requires specific origin but none provided
                continue

            if _is_active(excl, reference_date):
                active.append(excl)
            else:
                expired.append(excl)

        total_relief = sum(e.exclusion_rate_pct for e in active)

        return ExclusionLookupResult(
            active_exclusions=tuple(sorted(active, key=lambda e: e.exclusion_id)),
            expired_exclusions=tuple(sorted(expired, key=lambda e: e.exclusion_id)),
            total_exclusion_relief_pct=total_relief,
            has_active_exclusion=bool(active),
            has_expired_exclusion=bool(expired),
        )

    def check_by_overlay_type(
        self,
        overlay_type: str,
        hts_code: str,
        origin_country: str | None = None,
        *,
        reference_date: date | None = None,
    ) -> ExclusionLookupResult:
        """Check exclusions filtered by overlay type (section_232, section_301)."""
        full_result = self.check(hts_code, origin_country, reference_date=reference_date)

        active = tuple(e for e in full_result.active_exclusions if e.overlay_type == overlay_type)
        expired = tuple(e for e in full_result.expired_exclusions if e.overlay_type == overlay_type)
        total_relief = sum(e.exclusion_rate_pct for e in active)

        return ExclusionLookupResult(
            active_exclusions=active,
            expired_exclusions=expired,
            total_exclusion_relief_pct=total_relief,
            has_active_exclusion=bool(active),
            has_expired_exclusion=bool(expired),
        )


def _load_exclusions(path: Path) -> tuple[Exclusion, ...]:
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ()

    raw = payload.get("exclusions", [])
    if not isinstance(raw, list):
        return ()

    exclusions: list[Exclusion] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        exclusions.append(
            Exclusion(
                exclusion_id=str(entry.get("exclusion_id", "")),
                overlay_type=str(entry.get("overlay_type", "")),
                hts_codes=tuple(str(c) for c in entry.get("hts_codes", [])),
                product_description=str(entry.get("product_description", "")),
                origin_countries=tuple(
                    _normalize_country(c) for c in entry.get("origin_countries", [])
                ),
                exclusion_rate_pct=float(entry.get("exclusion_rate_pct", 0.0)),
                effective_date=str(entry.get("effective_date", "")),
                expiry_date=str(entry.get("expiry_date", "")),
                status=str(entry.get("status", "active")),
                proclamation_number=str(entry.get("proclamation_number", "")),
                federal_register_citation=str(entry.get("federal_register_citation", "")),
                requestor=str(entry.get("requestor", "")),
            )
        )
    return tuple(sorted(exclusions, key=lambda e: e.exclusion_id))


@lru_cache(maxsize=1)
def get_exclusion_tracker(data_path: str | None = None) -> ExclusionTracker:
    """Return a cached ExclusionTracker instance."""
    return ExclusionTracker(data_path)
