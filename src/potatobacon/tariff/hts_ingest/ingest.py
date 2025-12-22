from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.atom_utils import duty_rate_index

from .schema import TariffLine, TariffNote

REPO_ROOT = Path(__file__).resolve().parents[3].parent
DATA_DIR = REPO_ROOT / "data" / "hts_extract"
DEFAULT_LINES_PATH = DATA_DIR / "hts_lines_sample.jsonl"
DEFAULT_NOTES_PATH = DATA_DIR / "notes_sample.json"


@dataclass
class HTSIngestResult:
    atoms: List[PolicyAtom]
    duty_rates: Dict[str, float]
    lines: List[TariffLine]
    notes: List[TariffNote]


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: list[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def _load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_tariff_lines(path: Path | str = DEFAULT_LINES_PATH) -> List[TariffLine]:
    """Load tariff lines from a JSONL or CSV extract."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    if resolved.suffix.lower() == ".csv":
        raw_records = _load_csv(resolved)
    else:
        raw_records = _load_jsonl(resolved)

    lines = [TariffLine.from_payload(record) for record in raw_records]
    lines.sort(key=lambda line: (line.chapter, line.heading, line.hts_code, line.source_id))
    return lines


def load_tariff_notes(path: Path | str = DEFAULT_NOTES_PATH) -> List[TariffNote]:
    """Load chapter and heading notes from a JSON extract."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_notes: list[dict] = []
    raw_notes.extend(payload.get("chapter_notes") or [])
    raw_notes.extend(payload.get("heading_notes") or [])
    notes = [TariffNote.from_payload(item) for item in raw_notes]
    notes.sort(key=lambda note: (note.chapter, note.heading or "", note.note_id))
    return notes


def _line_metadata(line: TariffLine, note_lookup: Dict[str, TariffNote]) -> Dict[str, object]:
    note = note_lookup.get(line.note_id or "")
    metadata = {
        "hts_code": line.hts_code,
        "description": line.description,
        "duty_rate": line.duty_rate,
        "effective_date": line.effective_date,
        "source_ref": line.source_ref,
        "rate_applies": line.rate_applies,
        "citation": line.citation(),
    }
    if note:
        metadata["note_text"] = note.text
    return metadata


def _note_metadata(note: TariffNote) -> Dict[str, object]:
    return {
        "hts_code": note.heading or f"NOTE-{note.note_id}",
        "description": note.text,
        "duty_rate": 0.0,
        "effective_date": note.effective_date,
        "source_ref": note.source_ref,
        "rate_applies": note.rate_applies,
        "citation": note.citation(),
    }


def _line_to_atom(line: TariffLine, note_lookup: Dict[str, TariffNote]) -> PolicyAtom:
    metadata = _line_metadata(line, note_lookup)
    action = line.action
    if action is None and line.duty_rate is not None:
        duty_label = str(line.duty_rate).replace(".", "_")
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


def _note_to_atom(note: TariffNote) -> PolicyAtom:
    metadata = _note_metadata(note)
    action = note.action or f"note_{note.note_id}"
    return PolicyAtom(
        guard=list(note.guard_tokens),
        outcome={
            "modality": note.modality,
            "action": action,
            "subject": "classification_note",
            "jurisdiction": "US",
        },
        source_id=note.note_id,
        statute=note.statute,
        section=note.heading or note.note_id,
        text=note.text,
        modality=note.modality,
        action=action,
        rule_type="NOTE",
        atom_id=f"{note.note_id}_atom",
        metadata=metadata,
    )


def _build_atoms(lines: Iterable[TariffLine], notes: List[TariffNote]) -> List[PolicyAtom]:
    note_lookup = {note.note_id: note for note in notes}
    atoms: list[PolicyAtom] = []
    for line in lines:
        atoms.append(_line_to_atom(line, note_lookup))
    for note in notes:
        atoms.append(_note_to_atom(note))
    atoms.sort(key=lambda atom: (getattr(atom, "section", ""), atom.source_id))
    return atoms


_CACHE: Dict[Tuple[Path, Path], HTSIngestResult] = {}


def load_hts_policy_atoms(
    lines_path: Path | str = DEFAULT_LINES_PATH,
    notes_path: Path | str = DEFAULT_NOTES_PATH,
) -> HTSIngestResult:
    """Load the HTS policy atoms and duty rate index from structured extracts."""

    resolved_lines = Path(lines_path)
    resolved_notes = Path(notes_path)
    cache_key = (resolved_lines, resolved_notes)
    if cache_key in _CACHE:
        cached = _CACHE[cache_key]
        return HTSIngestResult(
            atoms=list(cached.atoms),
            duty_rates=dict(cached.duty_rates),
            lines=list(cached.lines),
            notes=list(cached.notes),
        )

    lines = load_tariff_lines(resolved_lines)
    notes = load_tariff_notes(resolved_notes)
    atoms = _build_atoms(lines, notes)
    duty_rates = duty_rate_index(atoms)
    result = HTSIngestResult(atoms=atoms, duty_rates=duty_rates, lines=lines, notes=notes)
    _CACHE[cache_key] = result
    return HTSIngestResult(
        atoms=list(result.atoms),
        duty_rates=dict(result.duty_rates),
        lines=list(result.lines),
        notes=list(result.notes),
    )


def load_policy_atoms() -> List[PolicyAtom]:
    """Convenience loader used by the tariff context manifest."""

    result = load_hts_policy_atoms()
    return list(result.atoms)
