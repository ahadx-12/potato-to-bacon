from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from potatobacon.proofs.canonical import compute_payload_hash
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


def _evidence_quality(fact_evidence: list | None) -> Tuple[int, int]:
    if not fact_evidence:
        return 0, 0
    snippets = sum(len(item.get("evidence", [])) for item in fact_evidence)
    return len(fact_evidence), snippets


LOAD_BEARING_FACTS: Dict[str, List[str]] = {
    ProductCategory.FOOTWEAR.value: [
        "product_category",
        "upper_material_textile",
        "outer_sole_material_rubber_or_plastics",
        "surface_contact_rubber_gt_50",
    ],
    ProductCategory.FASTENER.value: [
        "product_category",
        "product_type_chassis_bolt",
        "material_steel",
    ],
    ProductCategory.ELECTRONICS.value: [
        "product_category",
        "contains_pcb",
        "electronics_enclosure",
    ],
    ProductCategory.APPAREL_TEXTILE.value: [
        "product_category",
        "textile_knit",
        "fiber_cotton_dominant",
    ],
    ProductCategory.OTHER.value: ["product_category"],
}


def _init_category_bucket(category: str) -> Dict[str, Any]:
    return {
        "processed": 0,
        "ok": 0,
        "no_candidates": 0,
        "errors": 0,
        "savings_values": [],
        "risk_scores": [],
        "evidence": {
            "facts_with_evidence": 0,
            "total_facts": 0,
            "load_bearing_hits": 0,
            "load_bearing_total": 0,
        },
    }


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
    category_buckets: Dict[str, Dict[str, Any]] = defaultdict(dict)
    unknown_reasons: Dict[str, Dict[str, Any]] = {
        "no_keywords": {"count": 0, "examples": []},
        "conflicting_keywords": {"count": 0, "examples": []},
        "insufficient_material_info": {"count": 0, "examples": []},
        "bom_missing": {"count": 0, "examples": []},
    }
    origin_stats = {"origin_provided": 0, "requires_origin_data": 0, "ad_cvd_possible": 0}

    for entry in dataset:
        request = TariffSuggestRequestModel(
            sku_id=entry.get("sku_id"),
            description=entry["description"],
            bom_text=entry.get("bom_text"),
            bom_json=entry.get("bom_json"),
            bom_csv=entry.get("bom_csv"),
            declared_value_per_unit=entry.get("declared_value_per_unit"),
            annual_volume=entry.get("annual_volume"),
            law_context=entry.get("law_context") or law_context,
            top_k=top_k,
            seed=entry.get("seed") or seed,
            include_fact_evidence=include_evidence,
            origin_country=entry.get("origin_country"),
            export_country=entry.get("export_country"),
            import_country=entry.get("import_country"),
        )

        status_label = "ERROR"
        proof_retrieved: Dict[str, Any] | None = None
        best_payload: Dict[str, Any] | None = None
        best_suggestion = None
        annual_savings_value: float | None = None
        error_message: str | None = None
        evidence_ok: bool | None = None
        category_value: str = "unknown"
        category_bucket = category_buckets.setdefault(category_value, _init_category_bucket(category_value))
        response = None
        processed_logged = False

        try:
            response = suggest_tariff_optimizations(request)
            status_label = response.status
            if response.product_spec:
                category_value = response.product_spec.product_category.value
            elif entry.get("expected_category"):
                category_value = entry["expected_category"]
            category_bucket = category_buckets.setdefault(category_value, _init_category_bucket(category_value))
            if response.product_spec:
                category_counts[category_value] = category_counts.get(category_value, 0) + 1
            if not processed_logged:
                category_bucket["processed"] += 1
                processed_logged = True

            if include_evidence and response.fact_evidence is not None:
                dumped = [item.model_dump() for item in response.fact_evidence]
                fact_count, snippet_count = _evidence_quality(dumped)
                evidence_counter["facts_with_evidence"] += sum(1 for item in response.fact_evidence if item.evidence)
                evidence_counter["total_facts"] += fact_count
                evidence_counter["snippets"] += snippet_count
                category_bucket["evidence"]["facts_with_evidence"] += sum(
                    1 for item in response.fact_evidence if item.evidence
                )
                category_bucket["evidence"]["total_facts"] += fact_count
                load_bearing_keys = set(LOAD_BEARING_FACTS.get(category_value, []))
                category_bucket["evidence"]["load_bearing_total"] += len(load_bearing_keys)
                for fact in dumped:
                    if fact.get("fact_key") in load_bearing_keys and fact.get("evidence"):
                        category_bucket["evidence"]["load_bearing_hits"] += 1

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
                    category_bucket["risk_scores"].append(best_suggestion.risk_score)
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

            baseline = response.baseline_scenario or {}
            if baseline.get("requires_origin_data"):
                origin_stats["requires_origin_data"] += 1
            if baseline.get("ad_cvd_possible"):
                origin_stats["ad_cvd_possible"] += 1

        except Exception as exc:  # pragma: no cover - defensive logging
            error_message = str(exc)
            if category_value == "unknown" and entry.get("expected_category"):
                category_value = entry["expected_category"]
            category_bucket = category_buckets.setdefault(category_value, _init_category_bucket(category_value))
            if not processed_logged:
                category_bucket["processed"] += 1
                processed_logged = True

        if category_value in {ProductCategory.OTHER.value, "unknown"}:
            blob = " ".join(
                [
                    entry.get("description", ""),
                    entry.get("bom_text", ""),
                    entry.get("bom_csv", ""),
                    json.dumps(entry.get("bom_json"), sort_keys=True) if entry.get("bom_json") else "",
                ]
            ).lower()
            keyword_map = {
                "footwear": ["shoe", "sneaker", "boot"],
                "fastener": ["bolt", "fastener", "screw"],
                "electronics": ["pcb", "circuit", "chip", "sensor"],
                "apparel_textile": ["shirt", "hoodie", "jacket", "textile", "fabric"],
            }
            hits = {name: [kw for kw in kws if kw in blob] for name, kws in keyword_map.items()}
            hit_count = sum(1 for values in hits.values() if values)
            reason_key = "insufficient_material_info"
            if hit_count == 0:
                reason_key = "no_keywords"
            elif hit_count > 1:
                reason_key = "conflicting_keywords"
            elif not (entry.get("bom_text") or entry.get("bom_json") or entry.get("bom_csv")):
                reason_key = "bom_missing"
            reasons_bucket = unknown_reasons[reason_key]
            reasons_bucket["count"] += 1
            if len(reasons_bucket["examples"]) < 3:
                reasons_bucket["examples"].append(entry.get("sku_id"))

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

        if response and response.product_spec and response.product_spec.origin_country:
            origin_stats["origin_provided"] += 1

        if category_value in category_buckets:
            if status_label == "OK":
                category_buckets[category_value]["ok"] += 1
                if annual_savings_value is not None:
                    category_buckets[category_value]["savings_values"].append(annual_savings_value)
            elif status_label == "NO_CANDIDATES":
                category_buckets[category_value]["no_candidates"] += 1
            elif status_label == "ERROR":
                category_buckets[category_value]["errors"] += 1

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

    by_category: Dict[str, Dict[str, Any]] = {}
    for category, bucket in category_buckets.items():
        processed_count = bucket.get("processed", 0)
        ok_count = bucket.get("ok", 0)
        no_candidates_count = bucket.get("no_candidates", 0)
        error_count = bucket.get("errors", 0)
        by_category[category] = {
            "processed": processed_count,
            "ok": ok_count,
            "no_candidates": no_candidates_count,
            "errors": error_count,
            "ok_rate": (ok_count / processed_count * 100.0) if processed_count else 0.0,
            "avg_best_annual_savings": statistics.mean(bucket["savings_values"])
            if bucket.get("savings_values")
            else None,
            "avg_risk_score": statistics.mean(bucket["risk_scores"])
            if bucket.get("risk_scores")
            else None,
            "evidence_coverage_rate": (
                bucket["evidence"].get("facts_with_evidence", 0)
                / bucket["evidence"].get("total_facts", 0)
                * 100.0
                if bucket["evidence"].get("total_facts")
                else 0.0
            ),
            "load_bearing_evidence_rate": (
                bucket["evidence"].get("load_bearing_hits", 0)
                / (bucket["evidence"].get("load_bearing_total") or 0)
                * 100.0
                if bucket["evidence"].get("load_bearing_total")
                else 0.0
            ),
        }

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

        if not first_best and not second_best:
            stable = first.status == second.status == "NO_CANDIDATES"
        else:
            stable = (
                first_best is not None
                and second_best is not None
                and first_best.optimized_duty_rate == second_best.optimized_duty_rate
                and first_best.baseline_duty_rate == second_best.baseline_duty_rate
                and first_best.best_mutation == second_best.best_mutation
                and first_best.law_context == second_best.law_context
                and first_best.tariff_manifest_hash == second_best.tariff_manifest_hash
                and first_best.active_codes_optimized == second_best.active_codes_optimized
                and first_best.proof_payload_hash == second_best.proof_payload_hash
            )

            proof_first = proof_store.get_proof(first_best.proof_id) if first_best else None
            proof_second = proof_store.get_proof(second_best.proof_id) if second_best else None

            def _replay_hash(proof: Dict[str, Any] | None) -> str | None:
                if not proof:
                    return None
                return compute_payload_hash(proof)

            proof_hash_first = _replay_hash(proof_first)
            proof_hash_second = _replay_hash(proof_second)
            if proof_hash_first and proof_hash_second:
                stable = stable and proof_hash_first == proof_hash_second
            if proof_first and proof_hash_first:
                stable = stable and proof_first.get("proof_payload_hash") == proof_hash_first
            if proof_second and proof_hash_second:
                stable = stable and proof_second.get("proof_payload_hash") == proof_hash_second

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
        "evidence_coverage_rate": (
            evidence_counter["facts_with_evidence"] / evidence_counter["total_facts"] * 100.0
            if evidence_counter["total_facts"]
            else 0.0
        ),
        "load_bearing_evidence_rate": (
            sum(bucket["evidence"].get("load_bearing_hits", 0) for bucket in category_buckets.values())
            / sum(bucket["evidence"].get("load_bearing_total", 0) for bucket in category_buckets.values())
            * 100.0
            if sum(bucket["evidence"].get("load_bearing_total", 0) for bucket in category_buckets.values())
            else 0.0
        ),
        "by_category": by_category,
        "unknown_classification": unknown_reasons,
        "origin_stats": {
            "pct_origin_provided": (origin_stats["origin_provided"] / processed * 100.0) if processed else 0.0,
            "pct_requires_origin_data": (origin_stats["requires_origin_data"] / processed * 100.0)
            if processed
            else 0.0,
            "pct_ad_cvd_possible": (origin_stats["ad_cvd_possible"] / processed * 100.0)
            if processed
            else 0.0,
        },
        "determinism_passed": determinism_passed,
    }

    return {
        "results": results,
        "aggregates": aggregates,
        "determinism": {"passed": determinism_passed, "details": determinism_details},
    }
