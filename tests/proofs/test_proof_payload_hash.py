from copy import deepcopy

from potatobacon.proofs.canonical import compute_payload_hash


def _base_payload():
    return {
        "proof_id": "volatile-id",
        "created_at": "2024-01-01T00:00:00Z",
        "timestamp": "2024-01-01T00:00:01Z",
        "law_context": "TEST_CTX",
        "input": {
            "scenario": {"a": True, "b": False},
            "mutations": {"b": True},
        },
        "baseline": {
            "sat": True,
            "duty_rate": 5.0,
            "active_atoms": [
                {"source_id": "B", "atom_id": "2", "section": "b", "text": "bbb"},
                {"source_id": "A", "atom_id": "1", "section": "a", "text": "aaa"},
            ],
            "unsat_core": [],
        },
        "optimized": {
            "sat": True,
            "duty_rate": 3.0,
            "active_atoms": [
                {"source_id": "C", "atom_id": "3", "section": "c", "text": "ccc"},
                {"source_id": "A", "atom_id": "1", "section": "a", "text": "aaa"},
            ],
            "unsat_core": [],
        },
        "provenance_chain": [
            {"source_id": "Z", "section": "9", "text": "zzz"},
            {"source_id": "A", "section": "1", "text": "aaa"},
        ],
        "evidence_pack": {
            "fact_evidence": [
                {
                    "fact_key": "material",
                    "value": "textile",
                    "evidence": [
                        {
                            "source": "description",
                            "snippet": "textile upper",
                            "start": 0,
                            "end": 12,
                        }
                    ],
                },
                {
                    "fact_key": "material",
                    "value": "leather",
                    "evidence": [
                        {
                            "source": "description",
                            "snippet": "leather panel",
                            "start": 13,
                            "end": 26,
                        }
                    ],
                },
            ]
        },
        "compiled_facts": {"baseline": {"a": True}, "optimized": {"a": True, "b": True}},
    }


def test_proof_payload_hash_stability_against_ordering():
    payload = _base_payload()
    first = compute_payload_hash(payload)
    second = compute_payload_hash(payload)

    assert first == second

    reordered = deepcopy(payload)
    reordered["baseline"]["active_atoms"] = list(reversed(reordered["baseline"]["active_atoms"]))
    reordered["optimized"]["active_atoms"] = list(reversed(reordered["optimized"]["active_atoms"]))
    reordered["provenance_chain"] = list(reversed(reordered["provenance_chain"]))
    reordered["evidence_pack"]["fact_evidence"] = list(
        reversed(reordered["evidence_pack"]["fact_evidence"])
    )

    assert compute_payload_hash(reordered) == first

    mutated = deepcopy(payload)
    mutated["optimized"]["duty_rate"] = 1.5

    assert compute_payload_hash(mutated) != first
