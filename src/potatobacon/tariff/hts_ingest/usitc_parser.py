"""Parse USITC HTS records into the internal TariffLine schema.

The USITC provides records with fields like ``htsno``, ``description``,
``general`` (duty rate), ``special`` (preferential rates), ``indent``
(hierarchy level), and ``units``.  This module converts those into our
:class:`TariffLine` dataclass and auto-generates guard tokens from the
HTS hierarchy.

Duty rate parsing handles the common USITC formats:
  - "Free"
  - "2.5%"
  - "3.4¢/kg"
  - "6.5% + 2.1¢/kg"
  - Compound rates with ad valorem + specific components
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .guard_token_gen import generate_guard_tokens
from .schema import TariffLine

# ---------------------------------------------------------------------------
# Duty rate parsing
# ---------------------------------------------------------------------------
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CENTS_PER_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*¢\s*/\s*(\w+)")
_DOLLAR_PER_UNIT_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*/\s*(\w+)")


@dataclass
class ParsedDutyRate:
    """Structured representation of a USITC duty rate string."""

    raw: str
    ad_valorem_pct: float | None = None  # e.g. 2.5 for "2.5%"
    specific_amount: float | None = None  # e.g. 3.4 for "3.4¢/kg"
    specific_unit: str | None = None
    is_free: bool = False
    is_compound: bool = False
    is_unknown: bool = False


def parse_usitc_duty_rate(raw: str) -> ParsedDutyRate:
    """Parse a USITC duty rate string into structured form."""
    if not raw or not raw.strip():
        return ParsedDutyRate(raw=raw or "", is_unknown=True)

    cleaned = raw.strip()
    lower = cleaned.lower()

    if lower in ("free", "0", "0%", "0.0%"):
        return ParsedDutyRate(raw=raw, ad_valorem_pct=0.0, is_free=True)

    ad_valorem = None
    specific = None
    specific_unit = None

    pct_match = _PERCENT_RE.search(cleaned)
    if pct_match:
        ad_valorem = float(pct_match.group(1))

    cents_match = _CENTS_PER_UNIT_RE.search(cleaned)
    if cents_match:
        specific = float(cents_match.group(1)) / 100.0  # convert cents to dollars
        specific_unit = cents_match.group(2).lower()

    dollar_match = _DOLLAR_PER_UNIT_RE.search(cleaned)
    if dollar_match and specific is None:
        specific = float(dollar_match.group(1))
        specific_unit = dollar_match.group(2).lower()

    is_compound = ad_valorem is not None and specific is not None

    if ad_valorem is None and specific is None:
        return ParsedDutyRate(raw=raw, is_unknown=True)

    return ParsedDutyRate(
        raw=raw,
        ad_valorem_pct=ad_valorem,
        specific_amount=specific,
        specific_unit=specific_unit,
        is_free=False,
        is_compound=is_compound,
    )


def duty_rate_as_float(parsed: ParsedDutyRate) -> float | None:
    """Extract the ad valorem percentage as a float, or None."""
    if parsed.is_free:
        return 0.0
    if parsed.ad_valorem_pct is not None:
        return parsed.ad_valorem_pct
    # For specific-only rates we can't easily convert to %
    return None


# ---------------------------------------------------------------------------
# Special rate parsing
# ---------------------------------------------------------------------------
_SPECIAL_RATE_RE = re.compile(
    r"(?:Free|(\d+(?:\.\d+)?%))\s*\(([A-Z*+,\s]+)\)"
)


def parse_special_rates(raw: str) -> Dict[str, str]:
    """Parse USITC special rate string into {program_code: rate} dict.

    Example input: "Free (A*,AU,BH,CL,CO,D,E,IL,JO,KR,MA,OM,P,PA,PE,S,SG)"
    """
    if not raw or not raw.strip():
        return {}

    rates: Dict[str, str] = {}
    for match in _SPECIAL_RATE_RE.finditer(raw):
        rate = match.group(1) or "Free"
        codes = match.group(2)
        for code in codes.split(","):
            code = code.strip().rstrip("*+")
            if code:
                rates[code] = rate
    return rates


# ---------------------------------------------------------------------------
# HTS hierarchy helpers
# ---------------------------------------------------------------------------
def extract_chapter(htsno: str) -> str:
    """Extract chapter (first 2 digits) from an HTS number."""
    digits = re.sub(r"\D", "", htsno)
    return digits[:2] if len(digits) >= 2 else ""


def extract_heading(htsno: str) -> str:
    """Extract heading (first 4 digits) from an HTS number."""
    digits = re.sub(r"\D", "", htsno)
    return digits[:4] if len(digits) >= 4 else digits[:2]


def is_rate_line(htsno: str, indent: int) -> bool:
    """Check if a USITC record is an 8-digit rate line (legal tariff line).

    Rate lines have 8+ digits and indent >= 2.
    """
    digits = re.sub(r"\D", "", htsno)
    return len(digits) >= 8 and indent >= 1


def is_statistical_line(htsno: str) -> bool:
    """Check if this is a 10-digit statistical suffix line."""
    digits = re.sub(r"\D", "", htsno)
    return len(digits) >= 10


# ---------------------------------------------------------------------------
# USITC → TariffLine conversion
# ---------------------------------------------------------------------------
def usitc_record_to_tariff_line(
    record: Dict[str, Any],
    *,
    parent_descriptions: List[str] | None = None,
    effective_date: str = "2025-01-01",
) -> TariffLine | None:
    """Convert a USITC API record to our internal TariffLine format.

    Returns None for records that aren't rate lines (chapter/heading headers).
    """
    htsno = str(record.get("htsno") or "").strip()
    description = str(record.get("description") or "").strip()
    indent = int(record.get("indent") or 0)
    general_rate = str(record.get("general") or "").strip()

    if not htsno or not description:
        return None

    # Skip pure chapter/heading headers (indent 0) and section notes
    if indent == 0 and not general_rate:
        return None

    # Build source_id from HTS number
    digits = re.sub(r"\D", "", htsno)
    source_id = f"HTS_{digits}" if digits else f"HTS_{htsno.replace('.', '_')}"

    # Parse duty rate
    parsed_rate = parse_usitc_duty_rate(general_rate)
    duty_rate_float = duty_rate_as_float(parsed_rate)

    # Extract hierarchy
    chapter = extract_chapter(htsno)
    heading = extract_heading(htsno)

    # Parse special rates
    special_raw = str(record.get("special") or "").strip()
    special_rates = parse_special_rates(special_raw)

    # Generate guard tokens from HTS hierarchy + description
    all_descriptions = list(parent_descriptions or []) + [description]
    guard_tokens = generate_guard_tokens(
        htsno=htsno,
        description=description,
        parent_descriptions=parent_descriptions or [],
        indent=indent,
        chapter=chapter,
    )

    # Determine if this is a rate-bearing line
    rate_applies = duty_rate_float is not None

    return TariffLine(
        source_id=source_id,
        hts_code=htsno,
        description=description,
        duty_rate=duty_rate_float,
        effective_date=effective_date,
        chapter=chapter,
        heading=heading,
        note_id=None,
        source_ref=f"USITC HTS {effective_date[:4]}",
        guard_tokens=guard_tokens,
        jurisdiction="US",
        subject="import_duty",
        statute="HTSUS",
        rule_type="TARIFF",
        modality="OBLIGE",
        action=None,
        rate_applies=rate_applies,
    )


def parse_usitc_edition(
    records: List[Dict[str, Any]],
    *,
    effective_date: str = "2025-01-01",
    rate_lines_only: bool = True,
) -> List[TariffLine]:
    """Parse a full USITC edition into TariffLine objects.

    Tracks the hierarchy (parent descriptions at each indent level)
    and passes it through so guard tokens can be inherited.
    """
    lines: List[TariffLine] = []
    # Track hierarchy: indent level -> description
    hierarchy: Dict[int, str] = {}

    for record in records:
        htsno = str(record.get("htsno") or "").strip()
        description = str(record.get("description") or "").strip()
        indent = int(record.get("indent") or 0)

        if not htsno or not description:
            continue

        # Update hierarchy tracker
        hierarchy[indent] = description
        # Clear deeper levels
        for level in list(hierarchy.keys()):
            if level > indent:
                del hierarchy[level]

        # Build parent descriptions from hierarchy
        parent_descs = [
            hierarchy[i] for i in sorted(hierarchy.keys()) if i < indent
        ]

        # Skip non-rate lines if requested
        if rate_lines_only:
            digits = re.sub(r"\D", "", htsno)
            general = str(record.get("general") or "").strip()
            if len(digits) < 8 or not general:
                continue

        line = usitc_record_to_tariff_line(
            record,
            parent_descriptions=parent_descs,
            effective_date=effective_date,
        )
        if line is not None:
            lines.append(line)

    lines.sort(key=lambda l: (l.chapter, l.heading, l.hts_code, l.source_id))
    return lines
