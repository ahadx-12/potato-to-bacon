import pytest
from fastapi.testclient import TestClient

from potatobacon.law.solver_z3 import analyze_scenario
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.context_registry import load_atoms_for_context
from potatobacon.tariff.engine import apply_mutations, compute_duty_rate
from potatobacon.tariff.models import TariffScenario


@pytest.mark.usefixtures("system_client")
def test_proof_replay_integrity(system_client: TestClient):
    scenarios = [
        {
            "description": "Canvas sneaker with rubber sole",
            "declared_value_per_unit": 100.0,
            "annual_volume": 10_000,
        },
        {
            "description": "Chassis bolt fastener for EV frame",
            "declared_value_per_unit": 200.0,
            "annual_volume": 50_000,
        },
    ]

    for payload in scenarios:
        request = {**payload, "include_fact_evidence": True, "top_k": 3}
        response = system_client.post("/api/tariff/suggest", json=request)
        assert response.status_code == 200, response.text

        data = response.json()
        assert data["status"] == "OK"
        best = data["suggestions"][0]
        proof_id = best["proof_id"]

        proof_response = system_client.get(f"/v1/proofs/{proof_id}")
        assert proof_response.status_code == 200, proof_response.text
        proof_payload = proof_response.json()

        evidence_response = system_client.get(f"/v1/proofs/{proof_id}/evidence")
        assert evidence_response.status_code == 200, evidence_response.text

        atoms, context_meta = load_atoms_for_context(proof_payload["law_context"])
        baseline = TariffScenario(name="baseline", facts=proof_payload["input"]["scenario"])
        baseline_rate = compute_duty_rate(atoms, baseline)
        sat_baseline, active_baseline, _ = analyze_scenario(baseline.facts, atoms)

        optimized = apply_mutations(baseline, proof_payload["input"]["mutations"])
        optimized_rate = compute_duty_rate(atoms, optimized)
        sat_optimized, active_optimized, _ = analyze_scenario(optimized.facts, atoms)

        duty_codes_optimized = [
            atom.source_id for atom in active_optimized if atom.source_id in DUTY_RATES
        ]

        assert sat_baseline == proof_payload["baseline"]["sat"]
        assert sat_optimized == proof_payload["optimized"]["sat"]
        assert proof_payload["solver_result"] == ("SAT" if sat_baseline and sat_optimized else "UNSAT")

        assert pytest.approx(baseline_rate, rel=1e-6) == best["baseline_duty_rate"]
        assert pytest.approx(optimized_rate, rel=1e-6) == best["optimized_duty_rate"]
        assert duty_codes_optimized
        assert duty_codes_optimized[-1] == best["active_codes_optimized"][-1]
        assert best["tariff_manifest_hash"] == context_meta["manifest_hash"]
        assert proof_payload.get("tariff_manifest_hash") == context_meta["manifest_hash"]
        assert best["law_context"] == proof_payload["law_context"]
