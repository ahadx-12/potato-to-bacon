from __future__ import annotations

from fastapi import APIRouter, Depends

from potatobacon.api.context_helpers import unknown_law_context_error
from potatobacon.api.security import require_api_key
from potatobacon.tariff.context_registry import (
    DEFAULT_CONTEXT_ID,
    get_context_manifest,
    list_context_manifests,
    load_atoms_for_context,
)

router = APIRouter(
    prefix="/v1/law-contexts",
    tags=["law_contexts"],
    dependencies=[Depends(require_api_key)],
)


@router.get("")
def list_law_contexts():
    contexts = []
    for manifest in list_context_manifests(domain="tariff"):
        atoms, meta = load_atoms_for_context(manifest["context_id"])
        contexts.append(
            {
                "context_id": meta["context_id"],
                "jurisdiction": manifest.get("jurisdiction"),
                "effective_from": manifest.get("effective_from"),
                "effective_to": manifest.get("effective_to"),
                "description": manifest.get("description"),
                "manifest_hash": meta["manifest_hash"],
                "atoms_count": meta["atoms_count"],
            }
        )

    contexts.sort(key=lambda item: item["context_id"])
    return {
        "domain": "tariff",
        "default_context": DEFAULT_CONTEXT_ID,
        "contexts": contexts,
    }


@router.get("/{context_id}")
def get_law_context(context_id: str):
    try:
        manifest = get_context_manifest(context_id)
        _, meta = load_atoms_for_context(context_id)
    except KeyError as exc:
        raise unknown_law_context_error(context_id) from exc

    return {
        "manifest": manifest,
        "manifest_hash": meta["manifest_hash"],
        "atoms_count": meta["atoms_count"],
        "context_id": meta["context_id"],
        "domain": meta["domain"],
        "jurisdiction": manifest.get("jurisdiction"),
        "effective_from": manifest.get("effective_from"),
        "effective_to": manifest.get("effective_to"),
        "description": manifest.get("description"),
    }
