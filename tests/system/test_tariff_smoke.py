"""Smoke tests for CALE-TARIFF endpoints."""

import pytest


@pytest.mark.usefixtures("system_client")
def test_smoke_converse_optimize(system_client):
    payload = {
        "scenario": {
            "upper_material_textile": True,
            "outer_sole_material_rubber_or_plastics": True,
            "surface_contact_rubber_gt_50": True,
            "surface_contact_textile_gt_50": False,
            "felt_covering_gt_50": False,
        },
        "candidate_mutations": {
            "felt_covering_gt_50": [False, True],
        },
        "declared_value_per_unit": 100.0,
        "annual_volume": 10_000,
    }

    response = system_client.post("/api/tariff/optimize", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["status"] == "OPTIMIZED"
    assert pytest.approx(data["baseline_duty_rate"], rel=1e-6) == 37.5
    assert pytest.approx(data["optimized_duty_rate"], rel=1e-6) == 3.0
    assert pytest.approx(data["savings_per_unit"], rel=1e-6) == 34.5
    assert pytest.approx(data.get("savings_per_unit_rate"), rel=1e-6) == 34.5
    assert pytest.approx(data.get("savings_per_unit_value"), rel=1e-6) == 34.5
    assert pytest.approx(data.get("annual_savings_value"), rel=1e-6) == 345_000.0

    assert data["best_mutation"]["felt_covering_gt_50"] is True
    assert isinstance(data.get("proof_id"), str) and data["proof_id"]

    assert data.get("active_codes_baseline")
    assert data.get("active_codes_optimized")


@pytest.mark.usefixtures("system_client")
def test_smoke_tesla_sku_optimize(system_client):
    payload = {
        "sku_id": "TESLA_BOLT_X1",
        "description": "Chassis bolt for EV frame",
        "scenario": {
            "product_type_chassis_bolt": True,
            "material_steel": True,
            "material_aluminum": False,
        },
        "candidate_mutations": {
            "material_steel": [True, False],
            "material_aluminum": [False, True],
        },
        "declared_value_per_unit": 200.0,
        "annual_volume": 50_000,
    }

    response = system_client.post("/api/tariff/sku-optimize", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["sku_id"] == "TESLA_BOLT_X1"
    assert data["status"] == "OPTIMIZED"
    assert pytest.approx(data["baseline_duty_rate"], rel=1e-6) == 6.5
    assert pytest.approx(data["optimized_duty_rate"], rel=1e-6) == 2.5
    assert pytest.approx(data["savings_per_unit_rate"], rel=1e-6) == 4.0
    assert pytest.approx(data["savings_per_unit_value"], rel=1e-6) == 8.0
    assert pytest.approx(data["annual_savings_value"], rel=1e-6) == 400_000.0

    assert isinstance(data.get("best_mutation"), dict)
    assert isinstance(data.get("proof_id"), str) and data["proof_id"]


@pytest.mark.usefixtures("system_client")
def test_smoke_proof_retrieval(system_client):
    payload = {
        "scenario": {
            "upper_material_textile": True,
            "outer_sole_material_rubber_or_plastics": True,
            "surface_contact_rubber_gt_50": True,
            "surface_contact_textile_gt_50": False,
            "felt_covering_gt_50": False,
        },
        "candidate_mutations": {
            "felt_covering_gt_50": [False, True],
        },
        "declared_value_per_unit": 100.0,
        "annual_volume": 10_000,
    }

    optimize_response = system_client.post("/api/tariff/optimize", json=payload)
    assert optimize_response.status_code == 200, optimize_response.text

    proof_id = optimize_response.json()["proof_id"]

    proof_response = system_client.get(f"/v1/proofs/{proof_id}")
    assert proof_response.status_code == 200, proof_response.text

    proof = proof_response.json()
    assert proof["proof_id"] == proof_id
    assert "law_context" in proof
    assert proof.get("solver_result") in {"SAT", "UNSAT"}
    assert "input" in proof
    assert "baseline" in proof and "optimized" in proof
    assert proof["baseline"].get("active_atoms") is not None
    assert proof["optimized"].get("active_atoms") is not None
