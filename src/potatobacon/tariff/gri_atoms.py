from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Mapping

from potatobacon.law.solver_z3 import PolicyAtom


@dataclass(frozen=True)
class GRIRule:
    gri_id: str
    text: str


_GRI_RULES: List[GRIRule] = [
    GRIRule(
        gri_id="GRI_1",
        text="Classification shall be determined according to the terms of the headings and any relative section or chapter notes.",
    ),
    GRIRule(
        gri_id="GRI_2",
        text="Incomplete or unfinished goods are classified as the complete or finished goods if they have the essential character.",
    ),
    GRIRule(
        gri_id="GRI_3",
        text="When goods are prima facie classifiable under two headings, apply specificity, essential character, then last-in-order.",
    ),
    GRIRule(
        gri_id="GRI_4",
        text="Goods not classifiable under the above rules shall be classified under the heading appropriate to the goods to which they are most akin.",
    ),
    GRIRule(
        gri_id="GRI_5",
        text="Packaging and packing materials are classified with the goods if they are of a kind normally used for such goods.",
    ),
    GRIRule(
        gri_id="GRI_6",
        text="Classification at the subheading level shall be determined according to subheading terms and related notes.",
    ),
]


def gri_text_hash() -> str:
    """Return a deterministic hash of the canonical GRI text."""

    payload = "\n".join(rule.text for rule in _GRI_RULES)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_gri_atoms(
    *,
    revision_id: str,
    metadata_overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> List[PolicyAtom]:
    """Build global GRI atoms with deterministic metadata."""

    atoms: List[PolicyAtom] = []
    metadata_overrides = metadata_overrides or {}
    for index, rule in enumerate(_GRI_RULES, start=1):
        gri_fact = "gri_1_applies" if index == 1 else f"gri_{index}_applies"
        guard = []
        if index > 1:
            guard = ["¬gri_1_applies"]

        citation = {
            "statute": "GRI",
            "chapter": "GRI",
            "heading": None,
            "note_id": rule.gri_id,
            "source_ref": f"General Rules of Interpretation {index}",
            "revision_id": revision_id,
            "reference_id": f"{revision_id}::GRI::{rule.gri_id}",
        }
        metadata = {
            "citation": citation,
            "rate_applies": False,
        }
        metadata.update(metadata_overrides.get(rule.gri_id, {}))

        atoms.append(
            PolicyAtom(
                guard=guard,
                outcome={"modality": "OBLIGE", "action": gri_fact, "subject": "gri", "jurisdiction": "US"},
                source_id=rule.gri_id,
                statute="General Rules of Interpretation",
                section=rule.gri_id,
                text=rule.text,
                modality="OBLIGE",
                action=gri_fact,
                rule_type="GRI",
                atom_id=f"{rule.gri_id}_atom",
                metadata=metadata,
            )
        )

    return atoms


def gri_rule_ids() -> List[str]:
    return [rule.gri_id for rule in _GRI_RULES]


def gri_texts() -> Iterable[str]:
    return [rule.text for rule in _GRI_RULES]

