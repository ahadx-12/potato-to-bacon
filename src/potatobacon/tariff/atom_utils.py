from __future__ import annotations

from typing import Any, Dict, Iterable

from potatobacon.law.solver_z3 import PolicyAtom


def atom_provenance(atom: PolicyAtom, scenario_label: str | None = None) -> Dict[str, Any]:
    """Return a provenance entry enriched with ingestion metadata and citations."""

    metadata = getattr(atom, "metadata", {}) or {}
    citation = metadata.get("citation")
    summary_keys = ["hts_code", "description", "duty_rate", "effective_date", "source_ref"]
    summary = {key: metadata[key] for key in summary_keys if key in metadata}
    provenance = {
        "source_id": atom.source_id,
        "statute": getattr(atom, "statute", ""),
        "section": getattr(atom, "section", ""),
        "text": getattr(atom, "text", ""),
        "jurisdiction": atom.outcome.get("jurisdiction", ""),
    }
    if scenario_label:
        provenance["scenario"] = scenario_label
    if citation:
        provenance["citation"] = citation
    if summary:
        provenance["metadata"] = summary
    return provenance


def duty_rate_index(atoms: Iterable[PolicyAtom]) -> Dict[str, float]:
    """Derive a duty-rate lookup table from policy atoms with metadata."""

    rates: Dict[str, float] = {}
    for atom in atoms:
        metadata = getattr(atom, "metadata", {}) or {}
        rate_applies = metadata.get("rate_applies", True)
        duty_rate = metadata.get("duty_rate")
        if not rate_applies or duty_rate is None:
            continue
        try:
            rates[atom.source_id] = float(duty_rate)
        except (TypeError, ValueError):
            continue
    return rates
