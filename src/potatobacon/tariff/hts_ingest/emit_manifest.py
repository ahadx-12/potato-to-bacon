from __future__ import annotations

import json
from pathlib import Path

from potatobacon.tariff.context_registry import CONTEXTS_DIR, compute_context_hash

from .ingest import DEFAULT_LINES_PATH, DEFAULT_NOTES_PATH, load_hts_policy_atoms, load_policy_atoms

CONTEXT_ID = "HTS_US_2025_SLICE"


def build_manifest() -> dict:
    """Return the manifest payload for the HTS US 2025 slice."""

    return {
        "context_id": CONTEXT_ID,
        "domain": "tariff",
        "jurisdiction": "US",
        "effective_from": "2025-01-01",
        "effective_to": None,
        "description": "HTSUS 2025 slice generated from structured extract for chapters 64, 73, 85.",
        "loader": {
            "type": "python",
            "callable": "potatobacon.tariff.hts_ingest.ingest:load_policy_atoms",
        },
        "sources": [
            str(DEFAULT_LINES_PATH.relative_to(Path(__file__).resolve().parents[3].parent)),
            str(DEFAULT_NOTES_PATH.relative_to(Path(__file__).resolve().parents[3].parent)),
        ],
    }


def emit_manifest(out_dir: Path | None = None) -> Path:
    """Write the manifest to the contexts directory and return the path."""

    manifest = build_manifest()
    base_dir = out_dir or CONTEXTS_DIR
    manifest_dir = Path(base_dir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{CONTEXT_ID}.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest_path


def preview_manifest_hash() -> str:
    """Compute the manifest hash using the current ingest output."""

    atoms = load_policy_atoms()
    manifest = build_manifest()
    return compute_context_hash(manifest, atoms)


if __name__ == "__main__":  # pragma: no cover - manual tool
    path = emit_manifest()
    result = load_hts_policy_atoms()
    manifest_hash = compute_context_hash(build_manifest(), result.atoms)
    print(f"Wrote manifest to {path}")
    print(f"Atoms loaded: {len(result.atoms)}")
    print(f"Computed manifest hash: {manifest_hash}")
