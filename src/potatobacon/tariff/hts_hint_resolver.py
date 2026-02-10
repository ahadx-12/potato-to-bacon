"""HTS Code Hint Resolution for BOM rows with declared classifications.

When a BOM row includes a declared HTS code (importers often have their
current classification), this module:

1. Looks up the declared code in HTS_US_LIVE data
2. Extracts chapter/heading information
3. Scopes the optimization search to the same heading + adjacent headings
4. Falls back to full chapter-based search if the code isn't found
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from potatobacon.law.solver_z3 import PolicyAtom

logger = logging.getLogger(__name__)

# HTS code pattern: 4-10 digits, optionally with dots
_HTS_CODE_RE = re.compile(r"^(\d{4,10})(\.(\d{2,4}))?$")
_HTS_DOT_RE = re.compile(r"^(\d{4})\.(\d{2})(?:\.(\d{2,4}))?$")


def normalize_hts_code(code: str) -> Optional[str]:
    """Normalize an HTS code to a consistent format.

    Handles formats like: 7318.15.20, 73181520, 7318.15, 7318

    Returns:
        Normalized code string or None if invalid.
    """
    if not code:
        return None

    cleaned = code.strip().replace(" ", "")

    # Try dotted format first
    dot_match = _HTS_DOT_RE.match(cleaned)
    if dot_match:
        return cleaned

    # Try pure numeric
    digits = re.sub(r"[^\d]", "", cleaned)
    if len(digits) < 4:
        return None

    # Format as XXXX.XX.XXXX
    if len(digits) >= 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:]}"
    elif len(digits) >= 6:
        return f"{digits[:4]}.{digits[4:6]}"
    elif len(digits) >= 4:
        return digits[:4]

    return None


def extract_chapter(hts_code: str) -> Optional[str]:
    """Extract the 2-digit chapter number from an HTS code.

    Args:
        hts_code: Normalized HTS code.

    Returns:
        2-digit chapter string (e.g. "73") or None.
    """
    if not hts_code:
        return None
    digits = re.sub(r"[^\d]", "", hts_code)
    if len(digits) >= 2:
        return digits[:2]
    return None


def extract_heading(hts_code: str) -> Optional[str]:
    """Extract the 4-digit heading from an HTS code.

    Args:
        hts_code: Normalized HTS code.

    Returns:
        4-digit heading string (e.g. "7318") or None.
    """
    if not hts_code:
        return None
    digits = re.sub(r"[^\d]", "", hts_code)
    if len(digits) >= 4:
        return digits[:4]
    return None


def adjacent_headings(heading: str, range_size: int = 2) -> Set[str]:
    """Get adjacent HTS headings for scoped search.

    Args:
        heading: 4-digit heading string.
        range_size: How many headings above/below to include.

    Returns:
        Set of heading strings including the original and neighbors.
    """
    if not heading or len(heading) != 4:
        return {heading} if heading else set()

    try:
        heading_num = int(heading)
    except ValueError:
        return {heading}

    result: Set[str] = set()
    chapter = heading[:2]

    for offset in range(-range_size, range_size + 1):
        candidate = heading_num + offset
        # Stay within the same chapter (headings are XXYY where XX is chapter)
        candidate_str = f"{candidate:04d}"
        if candidate_str[:2] == chapter and candidate >= 0:
            result.add(candidate_str)

    return result


def resolve_hts_hint(
    declared_code: str,
    atoms: List[PolicyAtom],
    duty_rates: Dict[str, float],
) -> Tuple[Optional[PolicyAtom], Set[str], List[str]]:
    """Resolve a declared HTS code against loaded USITC atoms.

    Args:
        declared_code: The importer's declared HTS classification.
        atoms: Full list of loaded PolicyAtoms.
        duty_rates: Duty rate index from the context.

    Returns:
        Tuple of (matched_atom, scoped_headings, warnings) where:
        - matched_atom is the PolicyAtom matching the declared code (or None)
        - scoped_headings is the set of headings to scope the search to
        - warnings is a list of warning messages
    """
    warnings: List[str] = []
    normalized = normalize_hts_code(declared_code)

    if not normalized:
        warnings.append(f"Invalid HTS code format: {declared_code}")
        return None, set(), warnings

    chapter = extract_chapter(normalized)
    heading = extract_heading(normalized)

    if not chapter or not heading:
        warnings.append(f"Cannot extract chapter/heading from: {declared_code}")
        return None, set(), warnings

    # Look up the declared code in atoms
    # Match by source_id, section, or hts_code in metadata
    matched_atom: Optional[PolicyAtom] = None
    digits = re.sub(r"[^\d]", "", normalized)

    for atom in atoms:
        atom_hts = None
        # Check source_id
        atom_digits = re.sub(r"[^\d]", "", atom.source_id)
        if atom_digits == digits:
            matched_atom = atom
            break

        # Check section
        if atom.section:
            section_digits = re.sub(r"[^\d]", "", atom.section)
            if section_digits == digits:
                matched_atom = atom
                break

        # Check metadata
        metadata = getattr(atom, "metadata", None)
        if isinstance(metadata, dict):
            meta_hts = metadata.get("hts_code", "")
            if meta_hts:
                meta_digits = re.sub(r"[^\d]", "", str(meta_hts))
                if meta_digits == digits:
                    matched_atom = atom
                    break

    # Compute scoped headings
    scoped = adjacent_headings(heading)

    if matched_atom:
        logger.info(
            "HTS hint resolved: %s -> atom %s (rate=%s)",
            declared_code,
            matched_atom.source_id,
            duty_rates.get(matched_atom.source_id, "unknown"),
        )
    else:
        warnings.append(
            f"Declared HTS code {declared_code} not found in USITC data. "
            f"Falling back to chapter {chapter} search."
        )
        logger.warning(
            "HTS hint not found: %s — falling back to chapter %s",
            declared_code,
            chapter,
        )

    return matched_atom, scoped, warnings


def filter_atoms_by_headings(
    atoms: List[PolicyAtom],
    headings: Set[str],
    include_chapterless: bool = True,
) -> List[PolicyAtom]:
    """Filter atoms to those matching the given headings.

    Args:
        atoms: Full list of atoms.
        headings: Set of 4-digit heading strings to include.
        include_chapterless: Include atoms with no chapter guard tokens.

    Returns:
        Filtered list of atoms.
    """
    if not headings:
        return atoms

    filtered: List[PolicyAtom] = []
    for atom in atoms:
        atom_heading = None

        # Extract heading from source_id or section
        for ref in [atom.source_id, atom.section]:
            if ref:
                digits = re.sub(r"[^\d]", "", ref)
                if len(digits) >= 4:
                    atom_heading = digits[:4]
                    break

        # Also check metadata
        if not atom_heading:
            metadata = getattr(atom, "metadata", None)
            if isinstance(metadata, dict) and metadata.get("hts_code"):
                digits = re.sub(r"[^\d]", "", str(metadata["hts_code"]))
                if len(digits) >= 4:
                    atom_heading = digits[:4]

        if atom_heading and atom_heading in headings:
            filtered.append(atom)
        elif not atom_heading and include_chapterless:
            filtered.append(atom)

    logger.info(
        "HTS heading filter: %d -> %d atoms (headings=%s)",
        len(atoms),
        len(filtered),
        sorted(headings),
    )
    return filtered
