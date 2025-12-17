from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from potatobacon.proofs.store import get_default_store
from potatobacon.tariff.models import TariffSuggestRequestModel
from potatobacon.tariff.product_schema import ProductCategory
from potatobacon.tariff.suggest import suggest_tariff_optimizations


def _dataset_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "data" / "realworld_skus.json"


def _load_realworld_skus() -> List[Dict[str, Any]]:
    path = _dataset_path()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _annualized_savings(best, annual_volume: int | None) -> float | None:
    if best is None:
        return None
    if best.annual_savings_value is not None:
        return float(best.annual_savings_value)
    if best.savings_per_unit_value is not None and annual_volume is not None:
        return float(best.savings_per_unit_value) * annual_volume
    return float(best.savings_per_unit_value) if best.savings_per_unit_value is not None else None


def _normalized_proof_hash(proof: Dict[str, Any] | None) -> str | None:
    if not proof:
        return None
    sanitized = dict(proof)
    sanitized.pop("timestamp", None)
    sanitized.pop("proof_id", None)
    material = json.dumps(sanitized, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _evidence_quality(fact_evidence: list | None) -> Tuple[int, int]:
    if not fact_evidence:
        return 0, 0
    snippets = sum(len(item.get("evidence", [])) for item in fact_evidence)
    return len(fact_evidence), snippets


def run_readiness_eval(law_context: str | None = None, top_k: int = 3, include_evidence: bool = True) -> Dict[str, Any]:
    """Run deterministic readiness evaluation over the real-world SKU dataset."""

    dataset = _load_realworld_skus()
    proof_store = get_default_store()
    seed = 4242

    results: List[Dict[str, Any]] = []
    savings_values: List[float] = []
    risk_scores: List[int] = []
    risk_grades: List[str] = []
    category_counts: Dict[str, int] = {}
    evidence_counter = {"facts_with_evidence": 0, "total_facts": 0, "snippets": 0}

    for entry in dataset:
        request = TariffSuggestRequestModel(
            sku_id=entry.get("sku_id"),
            description=entry["description"],
            bom_text=entry.get("bom_text"),
            declared_value_per_unit=entry.get("declared_value_per_unit"),
            annual_volume=entry.get("annual_volume"),
            law_context=entry.get("law_context") or law_context,
            top_k=top_k,
            seed=entry.get("seed") or seed,
            include_fact_evidence=include_evidence,
        )

        status_label = "ERROR"
        proof_retrieved: Dict[str, Any] | None = None
        best_payload: Dict[str, Any] | None = None
        best_suggestion = None
        annual_savings_value: float | None = None
        error_message: str | None = None
        evidence_ok: bool | None = None

        try:
            response = suggest_tariff_optimizations(request)
            status_label = response.status
            if response.product_spec:
                category = response.product_spec.product_category.value
                category_counts[category] = category_counts.get(category, 0) + 1

            if include_evidence and response.fact_evidence is not None:
                fact_count, snippet_count = _evidence_quality(
                    [item.model_dump() for item in response.fact_evidence]
                )
                evidence_counter["facts_with_evidence"] += sum(
                    1 for item in response.fact_evidence if item.evidence
                )
                evidence_counter["total_facts"] += fact_count
                evidence_counter["snippets"] += snippet_count

            if response.suggestions:
                best_suggestion = response.suggestions[0]
                best_payload = best_suggestion.model_dump()
                annual_savings_value = _annualized_savings(
                    best_suggestion, request.annual_volume
                )
                if annual_savings_value is not None:
                    savings_values.append(annual_savings_value)
                if best_suggestion.risk_score is not None:
                    risk_scores.append(best_suggestion.risk_score)
                if best_suggestion.defensibility_grade:
                    risk_grades.append(best_suggestion.defensibility_grade)

                proof_retrieved = proof_store.get_proof(best_suggestion.proof_id)
                if proof_retrieved is None:
                    raise AssertionError("Proof missing from store")
                if include_evidence:
                    if response.fact_evidence is None:
                        raise AssertionError("Fact evidence missing when requested")
                    for item in response.fact_evidence:
                        if item.confidence < 0.0 or item.confidence > 1.0:
                            raise AssertionError("Evidence confidence out of bounds")
                    if (
                        response.product_spec
                        and response.product_spec.product_category != ProductCategory.OTHER
                    ):
                        evidence_ok = any(item.evidence for item in response.fact_evidence)
                        if not evidence_ok:
                            raise AssertionError("Expected evidence snippets for categorized SKU")
                    else:
                        evidence_ok = True

            if status_label == "OK" and not response.suggestions:
                status_label = "NO_CANDIDATES"

        except Exception as exc:  # pragma: no cover - defensive logging
            error_message = str(exc)

        results.append(
            {
                "sku_id": entry.get("sku_id"),
                "expected_category": entry.get("expected_category"),
                "expect_at_least_one_suggestion": entry.get("expect_at_least_one_suggestion"),
                "status": status_label,
                "best": best_payload,
                "annual_savings_value": annual_savings_value,
                "risk_score": best_suggestion.risk_score if best_suggestion else None,
                "risk_grade": best_suggestion.defensibility_grade if best_suggestion else None,
                "proof_id": best_suggestion.proof_id if best_suggestion else None,
                "proof_manifest_hash": (proof_retrieved or {}).get("tariff_manifest_hash"),
                "evidence_ok": evidence_ok,
                "error": error_message,
            }
        )

    processed = len(results)
    ok = len([r for r in results if r["status"] == "OK"])
    no_candidates = len([r for r in results if r["status"] == "NO_CANDIDATES"])
    errors = len([r for r in results if r["status"] == "ERROR"])
    ok_pct = (ok / processed * 100.0) if processed else 0.0

    savings_summary = None
    if savings_values:
        savings_summary = {
            "min": min(savings_values),
            "median": statistics.median(savings_values),
            "max": max(savings_values),
        }

    top_savings = sorted(
        [
            (
                r.get("annual_savings_value") or 0.0,
                r.get("sku_id"),
                r.get("best", {}).get("human_summary"),
            )
            for r in results
            if r.get("annual_savings_value") is not None
        ],
        key=lambda item: item[0],
        reverse=True,
    )

    top_entries = [
        {"sku_id": sku, "annual_savings_value": savings, "summary": summary}
        for savings, sku, summary in top_savings[:10]
    ]

    risk_avg = statistics.mean(risk_scores) if risk_scores else None
    risk_distribution: Dict[str, int] = {}
    for grade in risk_grades:
        risk_distribution[grade] = risk_distribution.get(grade, 0) + 1

    # Determinism check on the first five SKUs
    determinism_passed = True
    determinism_details: List[Dict[str, Any]] = []
    deterministic_subset = dataset[:5]

    for item in deterministic_subset:
        req_kwargs = dict(
            sku_id=item.get("sku_id"),
            description=item["description"],
            bom_text=item.get("bom_text"),
            declared_value_per_unit=item.get("declared_value_per_unit"),
            annual_volume=item.get("annual_volume"),
            law_context=item.get("law_context") or law_context,
            top_k=top_k,
            seed=item.get("seed") or seed,
            include_fact_evidence=include_evidence,
        )
        first = suggest_tariff_optimizations(TariffSuggestRequestModel(**req_kwargs))
        second = suggest_tariff_optimizations(TariffSuggestRequestModel(**req_kwargs))

        first_best = first.suggestions[0] if first.suggestions else None
        second_best = second.suggestions[0] if second.suggestions else None

        stable = (
            first_best is not None
            and second_best is not None
            and first_best.optimized_duty_rate == second_best.optimized_duty_rate
            and first_best.best_mutation == second_best.best_mutation
            and first_best.law_context == second_best.law_context
            and first_best.tariff_manifest_hash == second_best.tariff_manifest_hash
        )

        proof_hash_first = _normalized_proof_hash(
            proof_store.get_proof(first_best.proof_id) if first_best else None
        )
        proof_hash_second = _normalized_proof_hash(
            proof_store.get_proof(second_best.proof_id) if second_best else None
        )
        if proof_hash_first and proof_hash_second:
            stable = stable and proof_hash_first == proof_hash_second

        determinism_passed = determinism_passed and stable
        determinism_details.append(
            {
                "sku_id": item.get("sku_id"),
                "stable": stable,
                "law_context": first_best.law_context if first_best else None,
                "tariff_manifest_hash": first_best.tariff_manifest_hash if first_best else None,
            }
        )

    aggregates = {
        "processed": processed,
        "ok": ok,
        "no_candidates": no_candidates,
        "errors": errors,
        "ok_pct": ok_pct,
        "savings_summary": savings_summary,
        "risk_avg": risk_avg,
        "risk_distribution": risk_distribution,
        "top_savings": top_entries,
        "category_breakdown": category_counts,
        "evidence": evidence_counter,
        "determinism_passed": determinism_passed,
    }

    return {
        "results": results,
        "aggregates": aggregates,
        "determinism": {"passed": determinism_passed, "details": determinism_details},
    }
