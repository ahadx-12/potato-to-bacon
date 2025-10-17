"""Utilities for parsing human-entered "name: unit" mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import re


_LINE_RE = re.compile(r"[^\r\n]+")
_TRAILING_PUNCT = re.compile(r"[\s,;]+$")


@dataclass(slots=True)
class UnitsTextResult:
    """Result of :func:`parse_units_text`.

    Attributes
    ----------
    units:
        Mapping from symbol to the unit string as provided (trimmed).
    warnings:
        Human-readable warnings for lines that could not be parsed.
    """

    units: Dict[str, str]
    warnings: List[str]


def _strip_inline_comment(text: str) -> str:
    """Remove an inline ``#`` comment from ``text`` if present."""

    if "#" not in text:
        return text
    return text.split("#", 1)[0]


def _split_first_colon(text: str) -> Tuple[str, str] | None:
    """Split ``text`` on the first colon, returning ``(name, unit)``.

    Returns ``None`` when no colon is present.
    """

    idx = text.find(":")
    if idx == -1:
        return None
    return text[:idx], text[idx + 1 :]


def parse_units_text(units_text: str | None) -> UnitsTextResult:
    """Parse multiline ``"name: unit"`` text into a dictionary.

    Parameters
    ----------
    units_text:
        Raw text entered by the user. Each non-empty line should contain a
        variable name followed by a colon and the unit string. Comments starting
        with ``#`` are ignored. Trailing punctuation such as commas and
        semicolons is stripped.

    Returns
    -------
    UnitsTextResult
        A tuple containing the parsed mapping and a list of warnings for lines
        that could not be parsed. The parser never raises on malformed lines; it
        simply records a warning explaining the issue.
    """

    units: Dict[str, str] = {}
    warnings: List[str] = []

    if not units_text:
        return UnitsTextResult(units, warnings)

    for line_no, match in enumerate(_LINE_RE.finditer(units_text), start=1):
        raw_line = match.group(0)
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        pair = _split_first_colon(stripped)
        if pair is None:
            warnings.append(
                f"Line {line_no}: missing ':' — ignored: {raw_line.strip()!r}"
            )
            continue

        name, unit_part = pair
        name = name.strip()
        unit_part = _strip_inline_comment(unit_part).strip()
        unit_part = _TRAILING_PUNCT.sub("", unit_part)

        if not name:
            warnings.append(f"Line {line_no}: empty variable name.")
            continue

        if not unit_part:
            warnings.append(f"Line {line_no}: empty unit for {name!r}.")
            continue

        units[name] = unit_part

    return UnitsTextResult(units, warnings)


def iter_units_lines(units_text: str | None) -> Iterable[Tuple[str, str]]:
    """Yield ``(name, unit)`` pairs from ``units_text`` without warnings.

    This helper mirrors :func:`parse_units_text` but yields only the valid
    entries. It is primarily useful in tests where warnings are not needed.
    """

    result = parse_units_text(units_text)
    return result.units.items()

