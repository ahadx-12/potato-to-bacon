"""Per-step proof chain for tariff analysis audit trail.

Each intermediate step in the tariff analysis pipeline gets its own
SHA-256 hash, and the hashes are chained together so any tampering
with an intermediate step invalidates the final proof.

Chain structure::

    Step 1: BOM Input       → hash_bom
    Step 2: Fact Compilation → hash_facts (includes hash_bom)
    Step 3: Z3 Solve         → hash_solve (includes hash_facts)
    Step 4: Classification   → hash_classify (includes hash_solve)
    Step 5: Overlay          → hash_overlay (includes hash_classify)
    Step 6: Final Dossier    → hash_dossier (includes hash_overlay)

The final dossier hash verifies the entire chain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _canonical(obj: Any) -> str:
    """Deterministic JSON serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash(data: str) -> str:
    """SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class ProofStep:
    """A single step in the proof chain."""

    step_name: str
    step_index: int
    input_hash: str  # hash of this step's input
    output_hash: str  # hash of this step's output
    previous_hash: str  # chain link to previous step
    chain_hash: str  # hash(previous_hash + input_hash + output_hash)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "step_index": self.step_index,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "previous_hash": self.previous_hash,
            "chain_hash": self.chain_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ProofChain:
    """Builds a chained proof of each analysis step.

    Usage::

        chain = ProofChain()
        chain.add_step("bom_input", input_data=bom, output_data=bom)
        chain.add_step("fact_compilation", input_data=product_spec, output_data=facts)
        chain.add_step("z3_solve", input_data=facts, output_data=solver_result)
        chain.add_step("classification", input_data=solver_result, output_data=duty_result)
        chain.add_step("overlay", input_data=duty_result, output_data=effective_result)
        chain.add_step("dossier", input_data=effective_result, output_data=dossier)

        final_hash = chain.final_hash()
        chain_record = chain.to_dict()
    """

    def __init__(self) -> None:
        self.steps: List[ProofStep] = []
        self._genesis_hash = _hash("genesis")

    def add_step(
        self,
        step_name: str,
        *,
        input_data: Any,
        output_data: Any,
        metadata: Dict[str, Any] | None = None,
    ) -> ProofStep:
        """Add a step to the proof chain."""
        input_hash = _hash(_canonical(input_data))
        output_hash = _hash(_canonical(output_data))

        previous_hash = (
            self.steps[-1].chain_hash if self.steps else self._genesis_hash
        )

        chain_material = f"{previous_hash}:{input_hash}:{output_hash}"
        chain_hash = _hash(chain_material)

        step = ProofStep(
            step_name=step_name,
            step_index=len(self.steps),
            input_hash=input_hash,
            output_hash=output_hash,
            previous_hash=previous_hash,
            chain_hash=chain_hash,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def final_hash(self) -> str:
        """Return the final chain hash (or genesis if no steps)."""
        return self.steps[-1].chain_hash if self.steps else self._genesis_hash

    def verify(self) -> bool:
        """Verify the chain integrity by recomputing hashes."""
        previous = self._genesis_hash
        for step in self.steps:
            if step.previous_hash != previous:
                return False
            expected = _hash(f"{previous}:{step.input_hash}:{step.output_hash}")
            if step.chain_hash != expected:
                return False
            previous = step.chain_hash
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full chain for storage."""
        return {
            "chain_version": "1.0",
            "genesis_hash": self._genesis_hash,
            "final_hash": self.final_hash(),
            "step_count": len(self.steps),
            "verified": self.verify(),
            "steps": [step.to_dict() for step in self.steps],
        }
