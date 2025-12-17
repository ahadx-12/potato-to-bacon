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


@router.get("/{proof_id}/evidence")
def fetch_proof_evidence(proof_id: str):
    store = get_default_store()
    proof = store.get_proof(proof_id)
    if not proof:
        raise HTTPException(status_code=404, detail="Proof not found")
    pack = proof.get("evidence_pack") or {}
    return {
        "proof_id": proof_id,
        "law_context": proof.get("law_context"),
        "product_spec": pack.get("product_spec"),
        "compiled_facts": pack.get("compiled_facts"),
        "fact_evidence": pack.get("fact_evidence"),
    }
