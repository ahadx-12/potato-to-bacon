from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.duty_rate import DutyRate

REPO_ROOT = Path(__file__).resolve().parents[3].parent
DATA_DIR = REPO_ROOT / "data" / "hts_extract" / "full_chapters"


CHAPTER_PATHS: Dict[int, Path] = {
    39: DATA_DIR / "ch39.jsonl",
    84: DATA_DIR / "ch84.jsonl",
    87: DATA_DIR / "ch87.jsonl",
    90: DATA_DIR / "ch90.jsonl",
    94: DATA_DIR / "ch94.jsonl",
}

_PERCENT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)%\s*$")
_COMPOUND_RE = re.compile(
    r"^\s*\$(\d+(?:\.\d+)?)\s*/\s*([a-zA-Z]+)\s*\+\s*(\d+(?:\.\d+)?)%\s*$"
)


def parse_duty_rate(rate_str: str) -> DutyRate:
    if not rate_str:
        return DutyRate(type="unknown", raw=rate_str or "")
    normalized = rate_str.strip()
    if normalized.lower() == "free":
        return DutyRate(type="free", ad_valorem=0.0, raw=rate_str)
    percent_match = _PERCENT_RE.match(normalized)
    if percent_match:
        ad_valorem = float(percent_match.group(1)) / 100.0
        return DutyRate(type="ad_valorem", ad_valorem=ad_valorem, raw=rate_str)
    compound_match = _COMPOUND_RE.match(normalized)
    if compound_match:
        specific = float(compound_match.group(1))
        unit = compound_match.group(2).lower()
        ad_valorem = float(compound_match.group(3)) / 100.0
        return DutyRate(
            type="compound",
            specific=specific,
            specific_unit=unit,
            ad_valorem=ad_valorem,
            raw=rate_str,
        )
    return DutyRate(type="unknown", raw=rate_str)


def _normalize_record(record: Dict[str, object]) -> Dict[str, object]:
    chapter = str(record.get("chapter") or "")
    heading = str(record.get("heading") or "")
    subheading = str(record.get("subheading") or "")
    hts_code = str(record.get("hts_code") or "")
    description = str(record.get("description") or "")
    base_duty_rate = str(record.get("base_duty_rate") or "")
    unit_of_quantity = record.get("unit_of_quantity")
    special_rates = record.get("special_rates") or {}
    legal_notes = record.get("legal_notes") or []
    if isinstance(special_rates, list):
        special_rates = {str(item): "" for item in special_rates}
    if isinstance(legal_notes, str):
        legal_notes = [legal_notes]
    return {
        "chapter": chapter,
        "heading": heading,
        "subheading": subheading,
        "hts_code": hts_code,
        "description": description,
        "base_duty_rate": base_duty_rate,
        "unit_of_quantity": unit_of_quantity,
        "special_rates": {str(k): str(v) for k, v in dict(special_rates).items()},
        "legal_notes": [str(note) for note in legal_notes],
    }


def parse_hts_lines(source_path: Path) -> Iterator[dict]:
    with source_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            yield _normalize_record(record)


def _hts_sort_key(hts_code: str, description: str) -> tuple[int, str]:
    numeric = re.sub(r"\D", "", hts_code)
    return (int(numeric) if numeric else 0, description)


def _chapter_path(chapter_num: int, chapter_paths: Dict[int, Path] | None = None) -> Path:
    paths = chapter_paths or CHAPTER_PATHS
    if chapter_num not in paths:
        raise ValueError(f"Missing chapter fixture for {chapter_num}")
    return paths[chapter_num]


def ingest_chapter(chapter_num: int, source_path: Path | None = None) -> list[PolicyAtom]:
    resolved_path = source_path or _chapter_path(chapter_num)
    rows = list(parse_hts_lines(resolved_path))
    rows.sort(key=lambda row: _hts_sort_key(row["hts_code"], row["description"]))

    atoms: List[PolicyAtom] = []
    for row in rows:
        hts_code = row["hts_code"]
        atom_id = f"HTS_{hts_code.replace('.', '_')}"
        citation = {
            "source": "fixture",
            "chapter": row["chapter"],
            "heading": row["heading"],
            "hts_code": hts_code,
        }
        metadata = {
            "hts_code": hts_code,
            "description": row["description"],
            "base_duty_rate": row["base_duty_rate"],
            "special_rates": row["special_rates"],
            "unit_of_quantity": row["unit_of_quantity"],
            "chapter": row["chapter"],
            "heading": row["heading"],
            "legal_notes": row["legal_notes"],
            "citation": citation,
        }
        atoms.append(
            PolicyAtom(
                guard=[],
                outcome={"modality": "PERMIT", "action": atom_id, "subject": "hts_line", "jurisdiction": "US"},
                source_id=atom_id,
                statute="HTSUS",
                section=hts_code,
                text=row["description"],
                modality="PERMIT",
                action=atom_id,
                rule_type="HTS_LINE",
                atom_id=atom_id,
                metadata=metadata,
                hts_code=hts_code,
                description=row["description"],
                base_duty_rate=row["base_duty_rate"],
                special_rates=row["special_rates"],
                unit_of_quantity=row["unit_of_quantity"],
                chapter=row["chapter"],
                heading=row["heading"],
                legal_notes=row["legal_notes"],
                citation=citation,
            )
        )
    return atoms


def get_atom_by_hts(hts_code: str, chapter_paths: Dict[int, Path] | None = None) -> PolicyAtom:
    chapter = extract_chapter(hts_code)
    if chapter is None:
        raise ValueError(f"Invalid HTS code: {hts_code}")
    atoms = ingest_chapter(chapter, _chapter_path(chapter, chapter_paths))
    for atom in atoms:
        if atom.hts_code == hts_code:
            return atom
    raise KeyError(hts_code)


def extract_chapter(hts_code: str) -> int | None:
    digits = re.sub(r"\D", "", hts_code or "")
    if len(digits) < 2:
        return None
    return int(digits[:2])


def load_policy_atoms() -> List[PolicyAtom]:
    """Load all HTS policy atoms for the FULL context.

    Returns atoms from two sources, merged without duplicates:
    1. The base sample file (``hts_lines_sample.jsonl``) — these atoms carry
       Z3 guard tokens and duty rates, enabling scenario-based classification.
    2. The full chapter JSONL files (ch39/84/87/90/94) — these atoms carry
       rich descriptions for text search and GRI candidate matching.

    The base sample atoms take precedence on conflicts (same ``source_id``).
    """
    from potatobacon.tariff.hts_ingest.ingest import load_hts_policy_atoms

    # Phase 1: base sample atoms — have guard tokens, duty rates, Z3-evaluable
    base_result = load_hts_policy_atoms()
    merged: List[PolicyAtom] = list(base_result.atoms)
    seen_ids: set[str] = {atom.source_id for atom in merged}

    # Phase 2: full chapter atoms — rich descriptions for search, guard=[](empty)
    for chapter in sorted(CHAPTER_PATHS.keys()):
        for atom in ingest_chapter(chapter):
            if atom.source_id not in seen_ids:
                merged.append(atom)
                seen_ids.add(atom.source_id)

    return merged
