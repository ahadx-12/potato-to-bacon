import math


def _assert_arbitrage_dossier(dossier: dict, expected_jurisdictions: list[str], seed: int) -> None:
    golden = dossier["golden_scenario"]
    assert golden["jurisdictions"] == expected_jurisdictions

    metrics = dossier["metrics"]
    for key in ["value", "entropy", "kappa", "risk", "score"]:
        assert key in metrics

    assert metrics.get("value_components") and isinstance(metrics["value_components"], dict)
    assert metrics.get("risk_components") and isinstance(metrics["risk_components"], dict)

    score_components = metrics.get("score_components") or {}
    for key in ["value_term", "entropy_term", "risk_term", "alpha", "beta", "seed"]:
        assert key in score_components
    assert score_components["seed"] == seed

    value_term = score_components["value_term"]
    entropy_term = score_components["entropy_term"]
    risk_term = score_components["risk_term"]
    assert math.isclose(metrics["score"], value_term * entropy_term * risk_term, rel_tol=1e-6, abs_tol=1e-6)

    provenance = dossier.get("provenance_chain") or []
    assert len(provenance) >= 2
    for step in provenance[:2]:
        assert step["jurisdiction"]
        assert step["rule_id"]
        assert step.get("urn")
        assert step.get("citations")

    dependency_graph = dossier.get("dependency_graph") or {}
    assert dependency_graph.get("nodes")
    assert dependency_graph.get("edges")
    first_node = dependency_graph["nodes"][0]
    assert first_node.get("urn")
    assert first_node.get("citations") is not None

    proof_trace = dossier.get("proof_trace") or []
    assert proof_trace
    assert "risk_flags" in dossier

    assert isinstance(dossier.get("engine_version"), str)
    assert isinstance(dossier.get("manifest_hash"), str)


def test_arbitrage_sync_seed_reproducibility(authed_client, bulk_manifest):
    payload = {
        "manifest_hash": "latest",
        "request": {
            "jurisdictions": ["US", "IE", "KY"],
            "domain": "tax",
            "objective": "MAXIMIZE_NET_AFTER_TAX",
            "constraints": {"entity_type": ["corp"], "risk_tolerance": "medium"},
            "seed": 424242,
        },
    }

    first = authed_client.post("/api/law/arbitrage/hunt", json=payload)
    assert first.status_code == 200, first.text
    dossier_one = first.json()
    _assert_arbitrage_dossier(dossier_one, payload["request"]["jurisdictions"], payload["request"]["seed"])

    second = authed_client.post("/api/law/arbitrage/hunt", json=payload)
    assert second.status_code == 200, second.text
    dossier_two = second.json()
    _assert_arbitrage_dossier(dossier_two, payload["request"]["jurisdictions"], payload["request"]["seed"])

    assert dossier_one["golden_scenario"] == dossier_two["golden_scenario"]
    assert math.isclose(dossier_one["metrics"]["score"], dossier_two["metrics"]["score"], rel_tol=1e-6)

    alt_payload = payload.copy()
    alt_payload["request"] = dict(payload["request"], seed=424243)
    different = authed_client.post("/api/law/arbitrage/hunt", json=alt_payload)
    assert different.status_code == 200, different.text
    dossier_three = different.json()
    _assert_arbitrage_dossier(dossier_three, alt_payload["request"]["jurisdictions"], alt_payload["request"]["seed"])
