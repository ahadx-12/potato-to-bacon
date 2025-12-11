from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from potatobacon.api.security import require_api_key
from potatobacon.proofs.store import get_default_store

router = APIRouter(
    prefix="/v1/proofs",
    tags=["proofs"],
    dependencies=[Depends(require_api_key)],
)


@router.get("/{proof_id}")
def fetch_proof(proof_id: str):
    store = get_default_store()
    proof = store.get_proof(proof_id)
    if not proof:
        raise HTTPException(status_code=404, detail="Proof not found")
    return proof
