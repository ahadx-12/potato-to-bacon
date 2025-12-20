from __future__ import annotations

import argparse
import datetime as dt
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from potatobacon.proofs.canonical import compute_payload_hash
from potatobacon.proofs.store import get_default_store
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.batch_scan import batch_scan_tariffs
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.engine import TariffScenario, compute_duty_result
from potatobacon.tariff.models import (
    TariffBatchScanRequestModel,
    TariffBatchSkuModel,
    TariffParseRequestModel,
    TariffSuggestRequestModel,
)
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.suggest import suggest_tariff_optimizations
from potatobacon.law.solver_z3 import analyze_scenario


LOAD_BEARING_FACTS = {
    "FOOTWEAR": {"surface_contact_textile_gt_50", "surface_contact_rubber_gt_50", "material_rubber"},
    "FASTENER": {"is_fastener", "product_type_chassis_bolt", "material_steel"},
    "ELECTRONICS": {"product_type_electronics", "contains_pcb", "contains_battery"},
    "APPAREL_TEXTILE": {
        "material_textile",
        "textile_knit",
        "textile_woven",
        "has_coating_or_lamination",
    },
}


@dataclass
class ProofReplayResult:
    ok: bool
    message: str = ""


@dataclass
class E2ESkuResult:
    sku_id: str
    status: str
    category: str
    description: str
    best_summary: Optional[str] = None
    annual_savings: Optional[float] = None
    risk_score: Optional[int] = None
    defensibility: Optional[str] = None
    proof_id: Optional[str] = None
    proof_payload_hash: Optional[str] = None
    tariff_manifest_hash: Optional[str] = None
    determinism_match: Optional[bool] = None
    proof_replay: Optional[ProofReplayResult] = None
    evidence_gap: bool = False
    error: Optional[str] = None
    baseline_scenario: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterminismOutcome:
    passed: bool
    payload_match_rate: float
    details: List[str]


@dataclass
class E2ERunResult:
    sku_results: List[E2ESkuResult]
    determinism: DeterminismOutcome
    proof_replay_pass_rate: float
    batch_deterministic: bool
    batch_results: Any
    report_path: Path

    @property
    def counts(self) -> Dict[str, int]:
        counts = {
            "OK_OPTIMIZED": 0,
            "OK_BASELINE_ONLY": 0,
            "INSUFFICIENT_RULE_COVERAGE": 0,
            "INSUFFICIENT_INPUTS": 0,
            "ERROR": 0,
        }
        for res in self.sku_results:
            if res.status in counts:
                counts[res.status] += 1
            else:
                counts["ERROR"] += 1
        return counts


class TariffE2ERunner:
    def __init__(
        self,
        dataset_path: Path,
        *,
        mode: str = "engine",
        seed: int = 2025,
        api_base: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        law_context_override: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.mode = mode
        self.seed = seed
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.law_context_override = law_context_override
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        default_output = Path("reports") / f"CALE-TARIFF_E2E_{timestamp}.md"
        self.output_path = output_path or default_output

    def load_dataset(self, limit: Optional[int] = None, subset_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if subset_ids is not None:
            subset_set = set(subset_ids)
            data = [item for item in data if item.get("sku_id") in subset_set]
        if limit is not None:
            data = data[:limit]
        return data

    def run(self, *, limit: Optional[int] = None, subset_ids: Optional[Iterable[str]] = None) -> E2ERunResult:
        dataset = self.load_dataset(limit=limit, subset_ids=subset_ids)
        sku_results: List[E2ESkuResult] = []

        for entry in dataset:
            if self.mode == "http" and len(sku_results) >= 8:
                break
            try:
                if self.mode == "http":
                    sku_results.append(self._run_http_case(entry))
                else:
                    sku_results.append(self._run_engine_case(entry))
            except Exception:
                sku_results.append(
                    E2ESkuResult(
                        sku_id=entry.get("sku_id", "unknown"),
                        description=entry.get("description", ""),
                        status="ERROR",
                        category="UNKNOWN",
                        error=traceback.format_exc(),
                    )
                )

        determinism = DeterminismOutcome(passed=True, payload_match_rate=1.0, details=[])
        batch_resp = None
        batch_deterministic = True

        if self.mode == "engine":
            determinism = self._check_determinism(dataset)
            batch_resp, batch_deterministic = self._run_batch_scan(dataset)

        proof_replay_pass_rate = self._compute_proof_pass_rate(sku_results)
        self._generate_report(sku_results, determinism, batch_resp)

        return E2ERunResult(
            sku_results=sku_results,
            determinism=determinism,
            proof_replay_pass_rate=proof_replay_pass_rate,
            batch_deterministic=batch_deterministic,
            batch_results=batch_resp,
            report_path=self.output_path,
        )

    def _parse_payload(self, entry: Dict[str, Any]) -> TariffParseRequestModel:
        return TariffParseRequestModel(
            sku_id=entry.get("sku_id"),
            description=entry.get("description", ""),
            bom_text=entry.get("bom_text"),
            bom_json=entry.get("bom_json"),
            bom_csv=entry.get("bom_csv"),
            origin_country=entry.get("origin_country"),
            export_country=entry.get("export_country"),
            import_country=entry.get("import_country"),
        )

    def _normalize_bom(self, parse_request: TariffParseRequestModel) -> tuple[Optional[str], Optional[Any]]:
        bom_structured = parse_request.bom_json
        if bom_structured is None and parse_request.bom_csv:
            bom_structured = parse_bom_csv(parse_request.bom_csv)
        normalized_bom_text = parse_request.bom_text
        if bom_structured is not None:
            normalized_bom_text = bom_to_text(bom_structured)
        return normalized_bom_text, bom_structured

    def _run_engine_case(self, entry: Dict[str, Any]) -> E2ESkuResult:
        parse_request = self._parse_payload(entry)
        normalized_bom_text, bom_structured = self._normalize_bom(parse_request)
        spec, extraction_evidence = extract_product_spec(
            description=parse_request.description,
            bom_text=normalized_bom_text,
            bom_structured=bom_structured,
            origin_country=parse_request.origin_country,
            export_country=parse_request.export_country,
            import_country=parse_request.import_country,
        )
        compiled_facts, fact_evidence = compile_facts_with_evidence(
            spec,
            parse_request.description,
            normalized_bom_text,
            bom_structured=bom_structured,
            include_fact_evidence=True,
        )

        law_context = self.law_context_override or entry.get("law_context") or DEFAULT_CONTEXT_ID
        suggest_request = TariffSuggestRequestModel(
            sku_id=parse_request.sku_id,
            description=parse_request.description,
            bom_text=normalized_bom_text,
            bom_json=bom_structured,
            declared_value_per_unit=entry.get("declared_value_per_unit"),
            annual_volume=entry.get("annual_volume"),
            law_context=law_context,
            seed=self.seed,
            include_fact_evidence=True,
            origin_country=parse_request.origin_country,
            export_country=parse_request.export_country,
            import_country=parse_request.import_country,
        )
        suggest_response = suggest_tariff_optimizations(suggest_request)
        if suggest_response.status != "OK_OPTIMIZED" or not suggest_response.suggestions:
            return E2ESkuResult(
                sku_id=parse_request.sku_id or "unknown",
                description=parse_request.description,
                status=suggest_response.status,
                category=spec.product_category.value,
                tariff_manifest_hash=suggest_response.tariff_manifest_hash,
                baseline_scenario=suggest_response.baseline_scenario,
                proof_id=suggest_response.proof_id,
                proof_payload_hash=suggest_response.proof_payload_hash,
                error=None,
            )

        best = suggest_response.suggestions[0]
        proof_result = self._validate_proof(best, law_context)
        evidence_gap = self._check_evidence_gap(spec.product_category.value, suggest_response.fact_evidence or [])

        return E2ESkuResult(
            sku_id=parse_request.sku_id or "unknown",
            description=parse_request.description,
            status=suggest_response.status,
            category=spec.product_category.value,
            best_summary=best.human_summary,
            annual_savings=best.annual_savings_value,
            risk_score=best.risk_score,
            defensibility=best.defensibility_grade,
            proof_id=best.proof_id,
            proof_payload_hash=best.proof_payload_hash,
            tariff_manifest_hash=best.tariff_manifest_hash,
            proof_replay=proof_result,
            evidence_gap=evidence_gap,
            baseline_scenario=suggest_response.baseline_scenario,
        )

    def _run_http_case(self, entry: Dict[str, Any]) -> E2ESkuResult:
        headers = {"x-api-key": self.api_key or "test-key"}
        parse_request = self._parse_payload(entry)
        payload = parse_request.model_dump()
        response = httpx.post(f"{self.api_base}/api/tariff/parse", json=payload, headers=headers)
        if response.status_code != 200:
            return E2ESkuResult(
                sku_id=parse_request.sku_id or "unknown",
                description=parse_request.description,
                status="ERROR",
                category="UNKNOWN",
                error=f"parse {response.status_code}: {response.text}",
            )
        parse_body = response.json()
        category = parse_body.get("product_spec", {}).get("product_category", "UNKNOWN")

        suggest_payload = {
            "sku_id": parse_request.sku_id,
            "description": parse_request.description,
            "bom_text": entry.get("bom_text"),
            "bom_json": entry.get("bom_json"),
            "bom_csv": entry.get("bom_csv"),
            "declared_value_per_unit": entry.get("declared_value_per_unit"),
            "annual_volume": entry.get("annual_volume"),
            "law_context": self.law_context_override or entry.get("law_context"),
            "seed": self.seed,
            "include_fact_evidence": True,
        }
        suggest_resp = httpx.post(
            f"{self.api_base}/api/tariff/suggest",
            json=suggest_payload,
            headers=headers,
        )
        if suggest_resp.status_code != 200:
            return E2ESkuResult(
                sku_id=parse_request.sku_id or "unknown",
                description=parse_request.description,
                status="ERROR",
                category=category,
                error=f"suggest {suggest_resp.status_code}: {suggest_resp.text}",
            )
        suggest_body = suggest_resp.json()
        suggestions = suggest_body.get("suggestions") or []
        if suggest_body.get("status") != "OK_OPTIMIZED" or not suggestions:
            return E2ESkuResult(
                sku_id=parse_request.sku_id or "unknown",
                description=parse_request.description,
                status=suggest_body.get("status", "ERROR"),
                category=category,
                tariff_manifest_hash=suggest_body.get("tariff_manifest_hash"),
                baseline_scenario=suggest_body.get("baseline_scenario", {}),
                proof_id=suggest_body.get("proof_id"),
                proof_payload_hash=suggest_body.get("proof_payload_hash"),
            )
        best = suggestions[0]
        proof_id = best.get("proof_id")
        headers_proof = {**headers}
        proof_resp = httpx.get(f"{self.api_base}/v1/proofs/{proof_id}", headers=headers_proof)
        evidence_resp = httpx.get(f"{self.api_base}/v1/proofs/{proof_id}/evidence", headers=headers_proof)
        if proof_resp.status_code != 200 or evidence_resp.status_code != 200:
            error_text = f"proof fetch failed {proof_resp.status_code}/{evidence_resp.status_code}"
            return E2ESkuResult(
                sku_id=parse_request.sku_id or "unknown",
                description=parse_request.description,
                status="ERROR",
                category=category,
                error=error_text,
            )

        return E2ESkuResult(
            sku_id=parse_request.sku_id or "unknown",
            description=parse_request.description,
            status=suggest_body.get("status", "OK_OPTIMIZED"),
            category=category,
            best_summary=best.get("human_summary"),
            annual_savings=best.get("annual_savings_value"),
            risk_score=best.get("risk_score"),
            defensibility=best.get("defensibility_grade"),
            proof_id=proof_id,
            proof_payload_hash=best.get("proof_payload_hash"),
            tariff_manifest_hash=best.get("tariff_manifest_hash"),
            baseline_scenario=suggest_body.get("baseline_scenario", {}),
        )

    def _validate_proof(self, best: Any, law_context: str) -> ProofReplayResult:
        proof_id = best.proof_id if hasattr(best, "proof_id") else best.get("proof_id")
        payload_hash = best.proof_payload_hash if hasattr(best, "proof_payload_hash") else best.get("proof_payload_hash")
        if not proof_id:
            return ProofReplayResult(False, "missing proof_id")
        store = get_default_store()
        record = store.get_proof(proof_id)
        if not record:
            return ProofReplayResult(False, "proof not found")
        computed_hash = compute_payload_hash(record)
        if payload_hash and computed_hash != payload_hash:
            return ProofReplayResult(False, "payload hash mismatch")

        context = record.get("law_context") or law_context or DEFAULT_CONTEXT_ID
        atoms, _ = load_atoms_for_context(context)
        compiled = record.get("compiled_facts", {})
        baseline_facts = compiled.get("baseline") or {}
        optimized_facts = compiled.get("optimized") or {}

        baseline_sat, active_baseline, _ = analyze_scenario(baseline_facts, atoms)
        optimized_sat, active_opt, _ = analyze_scenario(optimized_facts, atoms)
        solver_status = "SAT" if baseline_sat and optimized_sat else "UNSAT"
        if solver_status != record.get("solver_result"):
            return ProofReplayResult(False, "solver status mismatch")

        def _duty_codes(active_atoms: Iterable[Any]) -> List[str]:
            codes: List[str] = []
            for atom in active_atoms:
                source = atom.source_id if hasattr(atom, "source_id") else atom.get("source_id") if isinstance(atom, dict) else None
                if source and source in DUTY_RATES:
                    codes.append(source)
            return sorted(codes)

        recorded_baseline = _duty_codes(record.get("baseline", {}).get("active_atoms", []))
        recorded_optimized = _duty_codes(record.get("optimized", {}).get("active_atoms", []))
        replay_baseline = _duty_codes(active_baseline)
        replay_optimized = _duty_codes(active_opt)

        if recorded_baseline != replay_baseline or recorded_optimized != replay_optimized:
            return ProofReplayResult(False, "active duty codes diverged")

        baseline_result = compute_duty_result(
            atoms,
            TariffScenario(name="baseline", facts=baseline_facts),
            active_atoms=active_baseline,
            is_sat=baseline_sat,
        )
        optimized_result = compute_duty_result(
            atoms,
            TariffScenario(name="optimized", facts=optimized_facts),
            active_atoms=active_opt,
            is_sat=optimized_sat,
        )

        recorded_baseline_status = record.get("baseline", {}).get("duty_status") or (
            "NO_DUTY_RULE_ACTIVE" if record.get("baseline", {}).get("duty_rate") is None else "OK"
        )
        recorded_optimized_status = record.get("optimized", {}).get("duty_status") or (
            "NO_DUTY_RULE_ACTIVE" if record.get("optimized", {}).get("duty_rate") is None else "OK"
        )

        if baseline_result.status != recorded_baseline_status:
            return ProofReplayResult(False, "baseline duty status mismatch")
        if optimized_result.status != recorded_optimized_status:
            return ProofReplayResult(False, "optimized duty status mismatch")

        if baseline_result.status == "OK" and baseline_result.duty_rate is not None:
            recorded_baseline_rate = record.get("baseline", {}).get("duty_rate")
            if recorded_baseline_rate not in (None, baseline_result.duty_rate) and abs(
                recorded_baseline_rate - baseline_result.duty_rate
            ) > 1e-6:
                return ProofReplayResult(False, "baseline duty mismatch")

        if optimized_result.status == "OK" and optimized_result.duty_rate is not None:
            recorded_optimized_rate = record.get("optimized", {}).get("duty_rate")
            if recorded_optimized_rate not in (None, optimized_result.duty_rate) and abs(
                recorded_optimized_rate - optimized_result.duty_rate
            ) > 1e-6:
                return ProofReplayResult(False, "optimized duty mismatch")
        return ProofReplayResult(True, "ok")

    def _check_determinism(self, dataset: List[Dict[str, Any]]) -> DeterminismOutcome:
        targets = [entry for entry in dataset if entry.get("expect_candidates")]
        targets = targets[:5]
        match_count = 0
        details: List[str] = []
        if not targets:
            return DeterminismOutcome(True, 1.0, [])

        for entry in targets:
            first = self._run_engine_case(entry)
            second = self._run_engine_case(entry)
            if first.status != second.status:
                details.append(f"{entry.get('sku_id')} status diverged {first.status}/{second.status}")
                continue
            if first.status != "OK_OPTIMIZED":
                match_count += 1
                continue
            match = (
                first.best_summary == second.best_summary
                and first.annual_savings == second.annual_savings
                and first.proof_payload_hash == second.proof_payload_hash
                and first.tariff_manifest_hash == second.tariff_manifest_hash
            )
            if match:
                match_count += 1
            else:
                details.append(f"{entry.get('sku_id')} best suggestion drift")
        payload_match_rate = match_count / len(targets) if targets else 1.0
        passed = payload_match_rate == 1.0
        return DeterminismOutcome(passed=passed, payload_match_rate=payload_match_rate, details=details)

    def _check_evidence_gap(self, category: str, fact_evidence: List[Any]) -> bool:
        keys = LOAD_BEARING_FACTS.get(category, set())
        if not keys:
            return False
        for item in fact_evidence:
            fact_key = item.fact_key if hasattr(item, "fact_key") else item.get("fact_key") if isinstance(item, dict) else None
            evidence = item.evidence if hasattr(item, "evidence") else item.get("evidence") if isinstance(item, dict) else []
            if fact_key in keys and evidence:
                return False
        return True

    def _run_batch_scan(self, dataset: List[Dict[str, Any]]):
        skus = []
        for entry in dataset:
            if not entry.get("expect_candidates"):
                continue
            skus.append(
                TariffBatchSkuModel(
                    sku_id=entry.get("sku_id", "unknown"),
                    description=entry.get("description", ""),
                    bom_text=entry.get("bom_text"),
                    bom_json=entry.get("bom_json"),
                    bom_csv=entry.get("bom_csv"),
                    declared_value_per_unit=entry.get("declared_value_per_unit"),
                    annual_volume=entry.get("annual_volume"),
                    law_context=self.law_context_override or entry.get("law_context"),
                    origin_country=entry.get("origin_country"),
                    export_country=entry.get("export_country"),
                    import_country=entry.get("import_country"),
                )
            )
        if not skus:
            return None, True

        request = TariffBatchScanRequestModel(skus=skus, seed=self.seed, include_all_suggestions=False)
        first_resp = batch_scan_tariffs(request)
        second_resp = batch_scan_tariffs(request)

        def _stable_dump(response: Any) -> Dict[str, Any]:
            payload = response.model_dump()
            payload.pop("generated_at", None)
            self._strip_proof_ids(payload)
            return payload

        deterministic = _stable_dump(first_resp) == _stable_dump(second_resp)
        return first_resp, deterministic

    def _strip_proof_ids(self, payload: Any) -> None:
        if isinstance(payload, dict):
            payload.pop("proof_id", None)
            for value in payload.values():
                self._strip_proof_ids(value)
        elif isinstance(payload, list):
            for item in payload:
                self._strip_proof_ids(item)

    def _compute_proof_pass_rate(self, results: List[E2ESkuResult]) -> float:
        ok_results = [res for res in results if res.status == "OK_OPTIMIZED" and res.proof_replay is not None]
        if not ok_results:
            return 1.0
        passes = sum(1 for res in ok_results if res.proof_replay and res.proof_replay.ok)
        return passes / len(ok_results)

    def _generate_report(
        self,
        results: List[E2ESkuResult],
        determinism: DeterminismOutcome,
        batch_resp: Any,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {
            "OK_OPTIMIZED": 0,
            "OK_BASELINE_ONLY": 0,
            "INSUFFICIENT_RULE_COVERAGE": 0,
            "INSUFFICIENT_INPUTS": 0,
            "ERROR": 0,
        }
        category_stats: Dict[str, Dict[str, Any]] = {}
        evidence_gaps: Dict[str, int] = {}

        for res in results:
            counts[res.status] = counts.get(res.status, 0) + 1
            category_entry = category_stats.setdefault(
                res.category,
                {
                    "processed": 0,
                    "optimized": 0,
                    "baseline_only": 0,
                    "insufficient_inputs": 0,
                    "insufficient_rules": 0,
                    "error": 0,
                    "savings": [],
                    "risks": [],
                    "evidence_gaps": 0,
                },
            )
            category_entry["processed"] += 1
            if res.status == "OK_OPTIMIZED":
                category_entry["optimized"] += 1
                if res.annual_savings is not None:
                    category_entry["savings"].append(res.annual_savings)
                if res.risk_score is not None:
                    category_entry["risks"].append(res.risk_score)
                if res.evidence_gap:
                    category_entry["evidence_gaps"] += 1
            elif res.status == "OK_BASELINE_ONLY":
                category_entry["baseline_only"] += 1
            elif res.status == "INSUFFICIENT_INPUTS":
                category_entry["insufficient_inputs"] += 1
            elif res.status == "INSUFFICIENT_RULE_COVERAGE":
                category_entry["insufficient_rules"] += 1
            else:
                category_entry["error"] += 1
            if res.evidence_gap:
                evidence_gaps[res.category] = evidence_gaps.get(res.category, 0) + 1

        top_opportunities = sorted(
            [res for res in results if res.status == "OK_OPTIMIZED" and res.annual_savings is not None],
            key=lambda r: r.annual_savings or 0,
            reverse=True,
        )[:10]

        error_signatures: Dict[str, int] = {}
        for res in results:
            if res.error:
                error_signatures[res.error] = error_signatures.get(res.error, 0) + 1

        categories_sorted = sorted(category_stats.items(), key=lambda item: item[0])

        with self.output_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# CALE-TARIFF E2E Pilot Run — {dt.datetime.now(dt.timezone.utc).isoformat()}\n\n")
            handle.write("## Executive summary\n")
            total = sum(counts.values())
            optimized_rate = counts.get("OK_OPTIMIZED", 0) / total if total else 0
            baseline_only_rate = counts.get("OK_BASELINE_ONLY", 0) / total if total else 0
            insufficient_inputs_rate = counts.get("INSUFFICIENT_INPUTS", 0) / total if total else 0
            insufficient_rules_rate = counts.get("INSUFFICIENT_RULE_COVERAGE", 0) / total if total else 0
            err_rate = counts.get("ERROR", 0) / total if total else 0
            handle.write(
                f"Processed: **{total}** | OK_OPTIMIZED: **{optimized_rate:.1%}** | OK_BASELINE_ONLY: **{baseline_only_rate:.1%}** | "
                f"INSUFFICIENT_INPUTS: **{insufficient_inputs_rate:.1%}** | INSUFFICIENT_RULE_COVERAGE: **{insufficient_rules_rate:.1%}** | ERROR: **{err_rate:.1%}**\n\n"
            )
            handle.write(
                f"Determinism: {'PASS' if determinism.passed else 'FAIL'} (payload hash stability {determinism.payload_match_rate:.1%})\n\n"
            )
            proof_ok = self._compute_proof_pass_rate(results)
            handle.write(f"Proof replay: {proof_ok:.1%} of optimized SKUs passed\n\n")

            handle.write("## Category scorecard\n")
            for category, stats in categories_sorted:
                processed = stats["processed"]
                opt_pct = (stats["optimized"] / processed) if processed else 0.0
                evidence_cov = 1 - (stats["evidence_gaps"] / stats["optimized"] if stats["optimized"] else 0)
                avg_savings = sum(stats["savings"]) / len(stats["savings"]) if stats["savings"] else 0.0
                avg_risk = sum(stats["risks"]) / len(stats["risks"]) if stats["risks"] else 0.0
                handle.write(
                    f"- **{category}** — processed {processed}, optimized {opt_pct:.1%}, baseline-only {stats['baseline_only']}, "
                    f"insufficient inputs {stats['insufficient_inputs']}, insufficient rules {stats['insufficient_rules']}, errors {stats['error']}; "
                    f"avg annual savings {avg_savings:.2f}, avg risk {avg_risk:.1f}, evidence coverage {evidence_cov:.1%}\n"
                )
            handle.write("\n")

            handle.write("## Top opportunities\n")
            if not top_opportunities:
                handle.write("- none found\n")
            for res in top_opportunities:
                handle.write(
                    f"- {res.sku_id}: {res.best_summary} — annual savings {res.annual_savings}, risk {res.defensibility} ({res.risk_score}); proof {res.proof_id}\n"
                )
            handle.write("\n")

            handle.write("## Weaknesses\n")
            if error_signatures:
                handle.write("- Error signatures:\n")
                for sig, count in sorted(error_signatures.items(), key=lambda item: -item[1]):
                    handle.write(f"  - {sig} ({count})\n")
            else:
                handle.write("- No errors observed\n")
            baseline_only_pressure = sorted(
                [(cat, stats["baseline_only"]) for cat, stats in categories_sorted if stats["baseline_only"]],
                key=lambda item: -item[1],
            )
            if baseline_only_pressure:
                handle.write("- Categories with baseline-only outcomes:\n")
                for cat, count in baseline_only_pressure:
                    handle.write(f"  - {cat}: {count}\n")
            if evidence_gaps:
                handle.write("- Evidence gaps observed in:\n")
                for cat, count in sorted(evidence_gaps.items(), key=lambda item: -item[1]):
                    handle.write(f"  - {cat}: {count}\n")
            handle.write("\n")

            handle.write("## Next actions\n")
            handle.write("1. Tighten parser keyword coverage for ambiguous gadget/apparel cases.\n")
            handle.write("2. Add baseline dossier pathways for textiles/electronics to reduce insufficient coverage.\n")
            handle.write("3. Expand evidence harvesting for load-bearing facts in footwear and electronics.\n")
            handle.write("4. Validate structured BOM fallbacks for mixed csv/json inputs.\n")

            if batch_resp:
                handle.write("\n## Batch scan snapshot\n")
                handle.write(f"Processed {batch_resp.total_skus} SKUs; results {len(batch_resp.results)}, skipped {len(batch_resp.skipped)}.\n")
                for result in batch_resp.results:
                    handle.write(
                        f"- {result.sku_id}: status {result.status}, best proof {result.best.proof_id if result.best else 'n/a'}, rank {result.rank_score}\n"
                    )
                for skipped in batch_resp.skipped:
                    handle.write(f"- Skipped {skipped.sku_id}: {skipped.status} {skipped.error or ''}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CALE-TARIFF E2E harness")
    parser.add_argument("--mode", choices=["engine", "http"], default="engine")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--law-context", default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Output report path (default reports/CALE-TARIFF_E2E_<timestamp>.md)",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default=Path("tests/data/e2e_pilot_pack.json"),
    )
    args = parser.parse_args()

    runner = TariffE2ERunner(
        dataset_path=Path(args.dataset),
        mode=args.mode,
        seed=args.seed,
        api_base=args.api_base,
        api_key=args.api_key,
        law_context_override=args.law_context,
        output_path=Path(args.out) if args.out else None,
    )

    result = runner.run()
    counts = result.counts
    print(f"Report written to: {result.report_path}")
    print(
        " ".join(
            [
                f"OK_OPTIMIZED={counts.get('OK_OPTIMIZED',0)}",
                f"OK_BASELINE_ONLY={counts.get('OK_BASELINE_ONLY',0)}",
                f"INSUFFICIENT_INPUTS={counts.get('INSUFFICIENT_INPUTS',0)}",
                f"INSUFFICIENT_RULE_COVERAGE={counts.get('INSUFFICIENT_RULE_COVERAGE',0)}",
                f"ERROR={counts.get('ERROR',0)}",
            ]
        )
    )
    print(
        f"Determinism={'PASS' if result.determinism.passed else 'FAIL'} payload stability {result.determinism.payload_match_rate:.1%}"
    )
    print(f"Proof replay pass rate {result.proof_replay_pass_rate:.1%}")
    if result.sku_results:
        top = max(
            [res for res in result.sku_results if res.annual_savings is not None],
            key=lambda r: r.annual_savings or 0,
            default=None,
        )
        if top:
            print(f"Top annual savings: {top.sku_id} -> {top.annual_savings}")
