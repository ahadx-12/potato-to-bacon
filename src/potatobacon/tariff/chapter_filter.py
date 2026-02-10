"""Chapter-based pre-filter for Z3 solver performance.

When the full USITC dataset (~12K atoms) is loaded, passing all atoms
to Z3 is expensive.  This module pre-filters atoms by chapter: given
the BOM's inferred chapters (via the fact mapper / vocabulary bridge),
only atoms whose guard tokens include a matching ``chapter_XX`` token
are passed to the solver.

Example:
    A steel bolt BOM expands to chapters 72, 73, 82, 83.
    The pre-filter reduces ~12K atoms to ~500, yielding significant
    Z3 solve-time improvements.

The filter is applied *before* ``check_sat`` as a function the solver
calls, not as a modification to the Z3 model itself.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Set, Tuple

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.fact_mapper import facts_to_guard_tokens

logger = logging.getLogger(__name__)

# Per-chapter atom cache: (context_id, chapter_set) -> filtered atoms
_chapter_atom_cache: Dict[Tuple[str, FrozenSet[str]], List[PolicyAtom]] = {}

_CHAPTER_RE = re.compile(r"^chapter_(\d{2})$")


def extract_chapters_from_facts(facts: Dict[str, Any]) -> Set[str]:
    """Extract chapter numbers from expanded facts.

    Scans the fact dictionary for ``chapter_XX`` keys that are True,
    returning the set of 2-digit chapter strings (e.g., ``{"72", "73"}``).

    Parameters
    ----------
    facts : dict
        Expanded fact dictionary (output of ``expand_facts()``).

    Returns
    -------
    Set of 2-digit chapter number strings.
    """
    chapters: Set[str] = set()
    for key, value in facts.items():
        if value is True:
            match = _CHAPTER_RE.match(key)
            if match:
                chapters.add(match.group(1))
    return chapters


def extract_chapters_from_atom(atom: PolicyAtom) -> Set[str]:
    """Extract chapter numbers from an atom's guard tokens.

    Parameters
    ----------
    atom : PolicyAtom
        The atom whose guard tokens to inspect.

    Returns
    -------
    Set of 2-digit chapter number strings found in the atom's guard.
    """
    chapters: Set[str] = set()
    for token in atom.guard:
        if token.startswith("\u00ac"):
            token = token[1:]
        match = _CHAPTER_RE.match(token)
        if match:
            chapters.add(match.group(1))
    return chapters


def filter_atoms_by_chapter(
    atoms: List[PolicyAtom],
    facts: Dict[str, Any],
    *,
    context_id: str = "",
    include_chapterless: bool = True,
) -> List[PolicyAtom]:
    """Filter atoms to only those matching the BOM's inferred chapters.

    This is the main pre-filter function, called before Z3 ``check_sat``.
    It does not modify the Z3 model; it reduces the input atom set.

    Parameters
    ----------
    atoms : list of PolicyAtom
        Full atom set (potentially ~12K atoms from USITC).
    facts : dict
        Expanded fact dictionary containing ``chapter_XX`` keys.
    context_id : str, optional
        Context identifier for caching. If empty, caching is disabled.
    include_chapterless : bool
        If True, atoms with no ``chapter_XX`` guard tokens (e.g., GRI
        rules, origin atoms, notes) are always included. Default True.

    Returns
    -------
    Filtered list of PolicyAtoms whose chapters overlap with the BOM's
    inferred chapters.
    """
    bom_chapters = extract_chapters_from_facts(facts)

    if not bom_chapters:
        logger.debug("No chapter tokens in facts; returning all %d atoms unfiltered", len(atoms))
        return atoms

    # Check cache
    cache_key = (context_id, frozenset(bom_chapters)) if context_id else None
    if cache_key and cache_key in _chapter_atom_cache:
        cached = _chapter_atom_cache[cache_key]
        logger.debug(
            "Chapter pre-filter cache hit: %d atoms for chapters %s",
            len(cached),
            sorted(bom_chapters),
        )
        return list(cached)

    filtered: List[PolicyAtom] = []
    for atom in atoms:
        atom_chapters = extract_chapters_from_atom(atom)

        if not atom_chapters:
            # Atom has no chapter guard tokens (e.g., GRI rules, notes)
            if include_chapterless:
                filtered.append(atom)
            continue

        # Keep atom if any of its chapters match the BOM's chapters
        if atom_chapters & bom_chapters:
            filtered.append(atom)

    # Populate cache
    if cache_key:
        _chapter_atom_cache[cache_key] = list(filtered)

    logger.info(
        "Chapter pre-filter: %d -> %d atoms (chapters=%s)",
        len(atoms),
        len(filtered),
        sorted(bom_chapters),
    )
    return filtered


def clear_chapter_cache() -> None:
    """Clear the per-chapter atom cache."""
    _chapter_atom_cache.clear()
