"""Wire USITC HTS data into the tariff context registry.

Provides a context loader that reads from the versioned USITC store
instead of hand-crafted JSONL files.  When a USITC edition is available,
it becomes the data source for the HTS_US_LIVE context.

Usage::

    atoms, meta = load_usitc_context()
    # These atoms have auto-generated guard tokens and can be used
    # directly by the Z3 solver and mutation engine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.atom_utils import duty_rate_index
from potatobacon.tariff.hts_ingest.usitc_parser import (
    duty_rate_as_float,
    parse_usitc_duty_rate,
    parse_usitc_edition,
)
from potatobacon.tariff.hts_ingest.versioned_store import VersionedHTSStore
from potatobacon.tariff.hts_ingest.ingest import _line_to_atom
from potatobacon.tariff.hts_ingest.schema import TariffNote

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[3].parent / "data" / "hts_extract" / "usitc"


def _tariff_line_to_policy_atom(line: Any) -> PolicyAtom:
    """Convert a TariffLine to a PolicyAtom, with USITC-specific metadata."""
    duty_rate_float = line.duty_rate

    metadata = {
        "hts_code": line.hts_code,
        "description": line.description,
        "duty_rate": duty_rate_float,
        "effective_date": line.effective_date,
        "source_ref": line.source_ref,
        "rate_applies": line.rate_applies,
        "citation": line.citation(),
    }

    action = line.action
    if action is None and duty_rate_float is not None:
        duty_label = str(duty_rate_float).replace(".", "_")
        action = f"duty_rate_{duty_label}"

    return PolicyAtom(
        guard=list(line.guard_tokens),
        outcome={
            "modality": line.modality,
            "action": action or line.source_id,
            "subject": line.subject,
            "jurisdiction": line.jurisdiction,
        },
        source_id=line.source_id,
        statute=line.statute,
        section=line.hts_code,
        text=line.description,
        modality=line.modality,
        action=action or line.source_id,
        rule_type=line.rule_type,
        atom_id=f"{line.source_id}_atom",
        metadata=metadata,
    )


def load_usitc_context(
    store_dir: Path | None = None,
    edition_id: str | None = None,
) -> Tuple[List[PolicyAtom], Dict[str, Any]]:
    """Load PolicyAtoms from a USITC edition in the versioned store.

    Parameters
    ----------
    store_dir : Path, optional
        Location of the versioned HTS store.
    edition_id : str, optional
        Specific edition to load.  If None, uses the current edition.

    Returns
    -------
    (atoms, metadata) tuple compatible with the context registry.
    """
    store = VersionedHTSStore(store_dir or DEFAULT_STORE_DIR)
    target_edition = edition_id or store.get_current_edition()

    if not target_edition:
        logger.warning("No USITC editions available, returning empty context")
        return [], {
            "context_id": "HTS_US_LIVE",
            "domain": "tariff",
            "atoms_count": 0,
            "duty_rates": {},
            "manifest_hash": "",
        }

    records = store.load_edition_records(target_edition)
    record_list = list(records.values())

    lines = parse_usitc_edition(record_list, rate_lines_only=True)
    atoms = [_tariff_line_to_policy_atom(line) for line in lines]

    duty_rates = duty_rate_index(atoms)

    edition_meta = {}
    for e in store.list_editions():
        if e.get("edition_id") == target_edition:
            edition_meta = e
            break

    metadata = {
        "context_id": "HTS_US_LIVE",
        "domain": "tariff",
        "jurisdiction": "US",
        "edition_id": target_edition,
        "description": f"USITC HTS Live ({target_edition})",
        "manifest_hash": edition_meta.get("sha256", ""),
        "atoms_count": len(atoms),
        "duty_rates": duty_rates,
        "record_count": len(record_list),
        "rate_line_count": len(lines),
    }

    logger.info(
        "Loaded USITC context %s: %d atoms from %d records",
        target_edition,
        len(atoms),
        len(record_list),
    )
    return atoms, metadata


def load_usitc_atoms() -> list[PolicyAtom]:
    """Load USITC atoms only (no metadata).

    This is the callable used by the context registry's Python loader.
    It returns a flat list of PolicyAtoms, compatible with
    ``_load_python_atoms`` in ``context_registry.py``.
    """
    atoms, _meta = load_usitc_context()
    return atoms
