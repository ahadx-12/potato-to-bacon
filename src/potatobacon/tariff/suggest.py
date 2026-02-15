from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from potatobacon.proofs.engine import record_tariff_proof
from potatobacon.cale.engine import CALEEngine
from potatobacon.tariff.bom_ingest import bom_to_text, parse_bom_csv
from potatobacon.tariff.candidate_search import generate_baseline_candidates
from potatobacon.tariff.context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context
from potatobacon.tariff.atoms_hts import DUTY_RATES
from potatobacon.tariff.engine import TariffScenario, _build_provenance, compute_duty_result
from potatobacon.tariff.fact_requirements import FactRequirementRegistry
from potatobacon.tariff.levers import LeverModel, applicable_levers, generate_candidate_levers
from potatobacon.tariff.mutation_generator import baseline_facts_from_profile, infer_product_profile
from potatobacon.tariff.models import (
    AutoClassificationResultModel,
    BaselineCandidateModel,
    NetSavings,
    TariffFeasibility,
    TariffOverlayResultModel,
    TariffSuggestRequestModel,
    TariffSuggestResponseModel,
    TariffSuggestionItemModel,
)
from potatobacon.tariff.reclassification_engine import (
    build_advisory_strategies,
    build_auto_classification_payload,
    build_reclassification_candidates,
)
from potatobacon.tariff.overlays import effective_duty_rate, evaluate_overlays
from potatobacon.tariff.sku_models import build_sku_metadata_snapshot
from potatobacon.tariff.normalizer import normalize_compiled_facts, validate_minimum_inputs
from potatobacon.tariff.parser import compile_facts_with_evidence, extract_product_spec
from potatobacon.tariff.risk import assess_tariff_risk
from potatobacon.law.solver_z3 import PolicyAtom, analyze_scenario


REPO_ROOT = Path(__file__).resolve().parents[3]
RULINGS_DIR = REPO_ROOT / "data" / "rulings"


def _compute_savings(
    baseline_rate: float | None,
    optimized_rate: float,
    declared_value_per_unit: float,
    annual_volume: int | None,
) -> tuple[float, float, float | None]:
    base = baseline_rate if baseline_rate is not None else optimized_rate
    rate_delta = base - optimized_rate
    savings_per_unit_value = rate_delta / 100.0 * declared_value_per_unit
    annual_savings_value = None
    if annual_volume is not None:
        annual_savings_value = savings_per_unit_value * annual_volume
    return rate_delta, savings_per_unit_value, annual_savings_value


def _feasibility_for_lever(lever: LeverModel | None) -> TariffFeasibility:
    if lever and lever.feasibility_profile:
        try:
            return TariffFeasibility(**lever.feasibility_profile)
        except Exception:
            pass
    return TariffFeasibility()


def _compute_net_savings(
    *,
    baseline_rate: float | None,
    optimized_rate: float | None,
    declared_value_per_unit: float,
    annual_volume: int | None,
    feasibility: TariffFeasibility,
) -> NetSavings:
    if baseline_rate is None or optimized_rate is None or annual_volume is None:
        return NetSavings()
    gross = (baseline_rate - optimized_rate) / 100.0 * declared_value_per_unit * annual_volume
    implementation_cost = feasibility.one_time_cost + feasibility.recurring_cost_per_unit * annual_volume
    first_year_adjustment = max(0.0, (365 - feasibility.implementation_time_days) / 365)
    first_year_savings = gross * first_year_adjustment
    net = first_year_savings - implementation_cost
    payback_months = None
    if gross > 0:
        monthly = gross / 12.0
        if monthly > 0:
            payback_months = implementation_cost / monthly
    return NetSavings(
        gross_duty_savings=gross,
        first_year_savings=first_year_savings,
        net_annual_savings=net,
        payback_months=payback_months,
    )


def _defensibility_grade(score: int) -> str:
    if score < 30:
        return "A"
    if score < 60:
        return "B"
    return "C"


def _degrade_grade(grade: str | None) -> str:
    if not grade:
        return "C"
    normalized = grade.upper()
    if normalized == "A":
        return "B"
    if normalized == "B":
        return "C"
    return normalized


def _apply_origin_precision_risk(
    grade: str,
    rvc_value: float | None,
    *,
    threshold: float = 60.0,
    margin: float = 2.0,
) -> str:
    if rvc_value is None:
        return grade
    if threshold <= rvc_value <= (threshold + margin):
        return "C"
    return grade


def _duty_atoms(active_atoms: Sequence[PolicyAtom], duty_rates: Mapping[str, float] | None = None) -> List[PolicyAtom]:
    rates = duty_rates or DUTY_RATES
    return [atom for atom in active_atoms if atom.source_id in rates]


def _normalize_hts_code(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.replace(".", "").replace(" ", "").strip().upper()


def _snapshot_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _load_ruling_snapshots() -> tuple[list[dict[str, Any]], list[str]]:
    rulings: list[dict[str, Any]] = []
    hashes: list[str] = []
    if not RULINGS_DIR.exists():
        return rulings, hashes
    for path in sorted(RULINGS_DIR.glob("*.jsonl")):
        hashes.append(_snapshot_hash(path))
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                rulings.append(payload)
    return rulings, hashes


def _rulings_for_hts(hts_code: str) -> list[dict[str, Any]]:
    normalized = _normalize_hts_code(hts_code)
    if not normalized:
        return []
    rulings, _ = _load_ruling_snapshots()
    matched: list[dict[str, Any]] = []
    for ruling in rulings:
        candidate = _normalize_hts_code(str(ruling.get("hts_code", "")))
        if not candidate:
            continue
        if normalized.startswith(candidate) or candidate.startswith(normalized):
            matched.append(ruling)
    return matched


def _precedent_context_for_hts(
    hts_code: str,
) -> tuple[dict[str, Any] | None, bool, list[str]]:
    rulings = _rulings_for_hts(hts_code)
    if not rulings:
        return None, False, []

    engine = CALEEngine()
    conflicts: list[dict[str, Any]] = []
    ccs_scores: list[float] = []
    conflicting_ids: list[str] = []
    for idx, left in enumerate(rulings):
        for right in rulings[idx + 1 :]:
            left_id = str(left.get("ruling_id") or left.get("id") or f"RULING_{idx}")
            right_id = str(right.get("ruling_id") or right.get("id") or f"RULING_{idx + 1}")
            left_rule = engine._ensure_rule(left, fallback_id=left_id)
            right_rule = engine._ensure_rule(right, fallback_id=right_id)
            analysis, components, metadata = engine._prepare_analysis(left_rule, right_rule)
            ccs_pragmatic = float(analysis.CCS_pragmatic)
            ccs_scores.append(ccs_pragmatic)
            if analysis.CI > 0.45 and ccs_pragmatic < 0.5:
                conflicts.append(
                    {
                        "ruling_ids": [left_id, right_id],
                        "ccs_pragmatic": ccs_pragmatic,
                        "conflict_intensity": float(analysis.CI),
                        "authority_balance": float(analysis.H),
                        "precedent_matches": metadata.get("precedent_matches", []),
                    }
                )
                conflicting_ids.extend([left_id, right_id])

    _, hashes = _load_ruling_snapshots()
    avg_ccs = sum(ccs_scores) / len(ccs_scores) if ccs_scores else 0.0
    authority_favors = avg_ccs >= 0.5
    context = {
        "hts_code": hts_code,
        "snapshot_hashes": hashes,
        "matched_rulings": [
            {
                "ruling_id": str(ruling.get("ruling_id") or ruling.get("id")),
                "title": ruling.get("title"),
                "hts_code": ruling.get("hts_code"),
                "year": ruling.get("enactment_year") or ruling.get("year"),
            }
            for ruling in rulings
        ],
        "conflicts": conflicts,
        "ccs_pragmatic_avg": avg_ccs,
        "authority_favors": authority_favors,
    }
    return context, bool(conflicts), sorted(set(conflicting_ids))


def _apply_precedent_signals(
    item: TariffSuggestionItemModel,
    atoms: Mapping[str, PolicyAtom],
) -> TariffSuggestionItemModel:
    hts_code = ""
    preferred_codes = []
    if item.target_candidate:
        preferred_codes.append(item.target_candidate)
    preferred_codes.extend(item.active_codes_optimized or item.active_codes_baseline)
    for code in preferred_codes:
        atom = atoms.get(code)
        metadata = getattr(atom, "metadata", {}) if atom else {}
        hts_code = str(metadata.get("hts_code") or "") if metadata else ""
        if hts_code:
            break

    context, has_conflict, conflicting_ids = _precedent_context_for_hts(hts_code)
    if context is None:
        return item

    payload = item.model_dump()
    payload["precedent_context"] = context
    if has_conflict:
        payload["defensibility_grade"] = _degrade_grade(payload.get("defensibility_grade"))
        reasons = list(payload.get("risk_reasons") or [])
        if conflicting_ids:
            reasons.append(f"Conflicting rulings: {', '.join(sorted(conflicting_ids))}")
        payload["risk_reasons"] = sorted(set(reasons))
    return TariffSuggestionItemModel(**payload)


@dataclass
class ScenarioEvaluation:
    sat: bool
    active_atoms: List[PolicyAtom]
    unsat_core: List[PolicyAtom]
    duty_atoms: List[PolicyAtom]
    duty_rate: float | None
    duty_status: str
    overlays: List[TariffOverlayResultModel]
    effective_duty_rate: float


DOCUMENTATION_LEVER_ID = "LEVER_DOC_SUBSTANTIATE_FACTS"


def _assumed_facts_for_candidate(atom: PolicyAtom, missing_facts: Sequence[str]) -> Dict[str, bool]:
    assumed: Dict[str, bool] = {}
    missing_set = set(missing_facts)
    for literal in atom.guard:
        negated = literal.startswith("¬")
        fact_key = literal[1:] if negated else literal
        if fact_key not in missing_set:
            continue
        assumed[fact_key] = not negated
    return assumed


def _documentation_lever_inputs(
    *,
    candidate: BaselineCandidateModel,
    atoms: Sequence[PolicyAtom],
    requirement_registry: FactRequirementRegistry,
) -> tuple[Dict[str, bool], list[str], list[str], list[str]]:
    atom_lookup = {atom.source_id: atom for atom in atoms}
    atom = atom_lookup.get(candidate.candidate_id)
    if atom is None:
        return {}, [], [], []

    assumed_facts = _assumed_facts_for_candidate(atom, candidate.missing_facts)
    fact_gaps = sorted(candidate.missing_facts)
    evidence_templates: list[str] = []
    why_needed: list[str] = []
    for fact_key in fact_gaps:
        requirement = requirement_registry.describe(fact_key)
        evidence_templates.extend(requirement.evidence_types)
        why_needed.append(requirement.render_question())
    evidence_templates = sorted({tmpl for tmpl in evidence_templates})
    return assumed_facts, fact_gaps, evidence_templates, sorted(set(why_needed))


def _documentation_feasibility() -> TariffFeasibility:
    return TariffFeasibility(
        one_time_cost=250.0,
        recurring_cost_per_unit=0.0,
        implementation_time_days=5,
        requires_recertification=False,
        supply_chain_risk="LOW",
    )


def _documentation_levers(
    *,
    baseline_candidates: Sequence[BaselineCandidateModel],
    atoms: Sequence[PolicyAtom],
    duty_rates: Mapping[str, float],
    baseline_eval: ScenarioEvaluation,
    baseline_facts: Mapping[str, object],
    baseline_rate: float | None,
    baseline_rate_raw: float | None,
    baseline_confidence: float,
    declared_value: float,
    annual_volume: int | None,
    law_context: str,
    context_meta: Mapping[str, Any],
    evidence_pack: Mapping[str, Any],
    overlay_context: Mapping[str, Any],
) -> list[TariffSuggestionItemModel]:
    requirement_registry = FactRequirementRegistry()
    documentation_levers: list[TariffSuggestionItemModel] = []
    for candidate in baseline_candidates:
        if not candidate.missing_facts:
            continue
        if baseline_rate is None:
            continue
        if baseline_rate is not None and candidate.duty_rate >= baseline_rate:
            continue
        assumed_facts, fact_gaps, evidence_templates, why_needed = _documentation_lever_inputs(
            candidate=candidate,
            atoms=atoms,
            requirement_registry=requirement_registry,
        )
        if not assumed_facts:
            continue

        mutated = deepcopy(baseline_facts)
        mutated.update(assumed_facts)
        mutated_normalized, _ = normalize_compiled_facts(mutated)

        optimized_eval = _evaluate_scenario(atoms, mutated_normalized, duty_rates, overlay_context=overlay_context)
        optimized_rate = optimized_eval.effective_duty_rate
        if not optimized_eval.sat or optimized_rate is None:
            continue
        if any(ov.stop_optimization for ov in optimized_eval.overlays):
            continue

        optimized_candidates = generate_baseline_candidates(mutated_normalized, atoms, duty_rates, max_candidates=1)
        optimized_confidence = optimized_candidates[0].confidence if optimized_candidates else baseline_confidence

        savings_rate, savings_value, annual_value = _compute_savings(
            baseline_rate=baseline_rate,
            optimized_rate=optimized_rate,
            declared_value_per_unit=declared_value,
            annual_volume=annual_volume,
        )
        if savings_rate <= 0 and optimized_confidence <= baseline_confidence + 0.2:
            continue

        feasibility_profile = _documentation_feasibility()
        net_savings = _compute_net_savings(
            baseline_rate=baseline_rate,
            optimized_rate=optimized_rate,
            declared_value_per_unit=declared_value,
            annual_volume=annual_volume,
            feasibility=feasibility_profile,
        )
        ranking_score = (
            net_savings.net_annual_savings
            if net_savings.net_annual_savings is not None
            else (annual_value if annual_value is not None else savings_value)
        )

        risk = assess_tariff_risk(
            baseline_facts=baseline_facts,
            optimized_facts=mutated_normalized,
            baseline_active_atoms=baseline_eval.active_atoms,
            optimized_active_atoms=optimized_eval.active_atoms,
            baseline_duty_rate=baseline_rate or optimized_rate,
            optimized_duty_rate=optimized_rate,
        )
        adjusted_risk_score = min(25, max(risk.risk_score, 10))
        adjusted_grade = _defensibility_grade(adjusted_risk_score)

        proof_handle = _record_proof(
            law_context=law_context,
            context_meta=context_meta,
            baseline_facts=baseline_facts,
            optimized_facts=mutated_normalized,
            baseline_eval=baseline_eval,
            optimized_eval=optimized_eval,
            lever=None,
            evidence_pack=evidence_pack,
            overlays={
                "baseline": [ov.model_dump() for ov in baseline_eval.overlays],
                "optimized": [ov.model_dump() for ov in optimized_eval.overlays],
            },
        )

        provenance_chain: List[Dict[str, Any]] = []
        provenance_chain.extend(_build_provenance(baseline_eval.duty_atoms, "baseline"))
        provenance_chain.extend(_build_provenance(optimized_eval.duty_atoms, "optimized"))
        provenance_chain.sort(
            key=lambda item: (
                item.get("source_id", ""),
                item.get("section", ""),
                item.get("text", ""),
                item.get("scenario", ""),
            )
        )

        documentation_levers.append(
            TariffSuggestionItemModel(
                human_summary="Substantiate missing facts to unlock lower-duty pathway",
                lever_category=DOCUMENTATION_LEVER_ID,
                lever_id=f"{DOCUMENTATION_LEVER_ID}:{candidate.candidate_id}",
                lever_feasibility="LOW",
                feasibility=feasibility_profile,
                evidence_requirements=list(evidence_templates),
                accepted_evidence_templates=list(evidence_templates),
                fact_gaps=fact_gaps,
                why_needed=why_needed,
                target_candidate=candidate.candidate_id,
                optimization_type="CONDITIONAL_OPTIMIZATION",
                baseline_duty_rate=baseline_rate_raw or baseline_eval.duty_rate or optimized_rate,
                optimized_duty_rate=optimized_eval.duty_rate if optimized_eval.duty_rate is not None else optimized_rate,
                baseline_effective_duty_rate=baseline_rate,
                optimized_effective_duty_rate=optimized_rate,
                savings_per_unit_rate=savings_rate,
                savings_per_unit_value=savings_value,
                annual_savings_value=annual_value,
                net_savings=net_savings,
                ranking_score=ranking_score,
                best_mutation=assumed_facts,
                classification_confidence=optimized_confidence,
                active_codes_baseline=sorted({atom.source_id for atom in baseline_eval.duty_atoms}),
                active_codes_optimized=sorted({atom.source_id for atom in optimized_eval.duty_atoms}),
                provenance_chain=provenance_chain,
                overlays={"baseline": baseline_eval.overlays, "optimized": optimized_eval.overlays},
                law_context=law_context,
                proof_id=proof_handle.proof_id,
                proof_payload_hash=proof_handle.proof_payload_hash,
                risk_score=adjusted_risk_score,
                defensibility_grade=adjusted_grade,
                risk_reasons=risk.risk_reasons,
                tariff_manifest_hash=context_meta["manifest_hash"],
                strategy_type="conditional_optimization",
                risk_level=_risk_level_for_score(adjusted_risk_score),
                required_actions=["Collect missing evidence and support selected classification pathway."],
                documentation_required=list(evidence_templates),
                confidence_level=optimized_confidence,
                implementation_difficulty="low",
            )
        )

    documentation_levers.sort(key=lambda item: item.optimized_duty_rate)
    return documentation_levers


def _evaluate_scenario(
    atoms: Sequence[PolicyAtom],
    facts: Mapping[str, object],
    duty_rates: Mapping[str, float] | None = None,
    *,
    overlay_context: Mapping[str, Any] | None = None,
) -> ScenarioEvaluation:
    rates = duty_rates or DUTY_RATES
    sat, active_atoms, unsat_core = analyze_scenario(facts, atoms)
    scenario = TariffScenario(name="scenario", facts=dict(facts))
    duty_result = compute_duty_result(
        atoms, scenario, active_atoms=list(active_atoms), is_sat=sat, duty_rates=rates
    )
    duty_atoms = duty_result.active_atoms or _duty_atoms(active_atoms, rates)
    duty_rate = duty_result.duty_rate
    if sat and duty_atoms:
        ranked = sorted(
            duty_atoms,
            key=lambda atom: (
                float(rates[atom.source_id]),
                -len(atom.guard),
                atom.source_id,
            ),
        )
        duty_atoms = ranked
        duty_rate = float(rates[ranked[0].source_id])
    overlay_ctx = overlay_context or {}
    overlays = evaluate_overlays(
        facts=facts,
        active_codes=[atom.source_id for atom in duty_atoms],
        origin_country=overlay_ctx.get("origin_country"),
        import_country=overlay_ctx.get("import_country"),
        hts_code=overlay_ctx.get("hts_code"),
    )
    effective_rate = effective_duty_rate(duty_rate, overlays)
    return ScenarioEvaluation(
        sat=sat,
        active_atoms=list(active_atoms),
        unsat_core=list(unsat_core),
        duty_atoms=list(duty_atoms),
        duty_rate=duty_rate,
        duty_status=duty_result.status,
        overlays=overlays,
        effective_duty_rate=effective_rate,
    )


def _select_baseline_rate(
    baseline_eval: ScenarioEvaluation,
    baseline_candidates: Sequence[BaselineCandidateModel],
    duty_rates: Mapping[str, float],
) -> tuple[float | None, float | None]:
    if getattr(baseline_eval, "duty_atoms", None):
        ranked_atoms = sorted(
            baseline_eval.duty_atoms,
            key=lambda atom: (float(duty_rates.get(atom.source_id, 999.0)), atom.source_id),
        )
        top_atom = ranked_atoms[0]
        candidate_lookup = {cand.candidate_id: cand for cand in baseline_candidates}
        candidate = candidate_lookup.get(top_atom.source_id)
        confidence = candidate.confidence if candidate else None
        rate = float(duty_rates.get(top_atom.source_id, baseline_eval.duty_rate))
        return rate, confidence

    eligible = [
        cand
        for cand in baseline_candidates
        if not cand.missing_facts and cand.compliance_flags.get("guard_satisfied", False)
    ]
    eligible.sort(
        key=lambda cand: (
            cand.duty_rate,
            -len(cand.active_codes),
            -(cand.confidence or 0.0),
            cand.candidate_id,
        )
    )
    if eligible:
        best = eligible[0]
        return best.duty_rate, best.confidence

    fallback_confidence = baseline_candidates[0].confidence if baseline_candidates else None
    return baseline_eval.duty_rate, fallback_confidence


def _sort_key(item: TariffSuggestionItemModel, annual_volume: int | None, index: int) -> tuple:
    net_value = item.net_savings.net_annual_savings if item.net_savings else None
    primary = net_value if net_value is not None else (
        item.annual_savings_value if annual_volume is not None else item.savings_per_unit_value
    )
    primary_value = primary if primary is not None else item.savings_per_unit_rate
    difficulty_rank = {"low": 0, "medium": 1, "high": 2}.get((item.implementation_difficulty or "").lower(), 3)
    risk_score = item.risk_score if item.risk_score is not None else 999
    confidence = item.classification_confidence if item.classification_confidence is not None else 0.0
    return (-primary_value, difficulty_rank, risk_score, -confidence, item.lever_id or "", index)


def _risk_level_for_score(score: int | None) -> str:
    value = score if score is not None else 60
    if value <= 30:
        return "low_risk"
    if value <= 60:
        return "medium_risk"
    return "high_risk"


def _strategy_type_for_lever(lever: LeverModel) -> str:
    mutation_keys = set(lever.mutation.keys())
    if {"origin_country", "export_country"} & mutation_keys:
        return "origin_shift"
    if lever.lever_type == "DOCUMENTATION":
        return "conditional_optimization"
    return "product_modification"


def _grade_rank(value: str | None) -> int:
    lookup = {"A": 0, "B": 1, "C": 2}
    return lookup.get((value or "C").upper(), 2)


def _passes_constraints(item: TariffSuggestionItemModel, request: TariffSuggestRequestModel) -> bool:
    tolerance_rank = _grade_rank(request.risk_tolerance)
    item_rank = _grade_rank(item.defensibility_grade)
    if item_rank > tolerance_rank:
        return False

    if item.net_savings and item.net_savings.net_annual_savings is not None:
        if item.net_savings.net_annual_savings < request.min_net_savings:
            return False
    elif request.min_net_savings > 0:
        return False

    if request.max_payback_months is not None:
        if not item.net_savings or item.net_savings.payback_months is None:
            return False
        if item.net_savings.payback_months > request.max_payback_months:
            return False

    if not request.allow_recertification and item.feasibility and item.feasibility.requires_recertification:
        return False

    return True


def _record_proof(
    *,
    law_context: str,
    context_meta: Mapping[str, Any],
    baseline_facts: Mapping[str, object],
    optimized_facts: Mapping[str, object],
    baseline_eval: ScenarioEvaluation,
    optimized_eval: ScenarioEvaluation,
    lever: LeverModel | None,
    evidence_pack: Mapping[str, Any] | None = None,
    overlays: Mapping[str, Any] | None = None,
):
    baseline_active = baseline_eval.active_atoms
    baseline_unsat = baseline_eval.unsat_core
    baseline_rate = baseline_eval.duty_rate
    optimized_active = optimized_eval.active_atoms
    optimized_unsat = optimized_eval.unsat_core
    optimized_rate = optimized_eval.duty_rate
    provenance_chain: List[Dict[str, Any]] = []
    provenance_chain.extend(_build_provenance(_duty_atoms(baseline_active), "baseline"))
    provenance_chain.extend(_build_provenance(_duty_atoms(optimized_active), "optimized"))
    provenance_chain.sort(
        key=lambda item: (
            item.get("source_id", ""),
            item.get("section", ""),
            item.get("text", ""),
            item.get("scenario", ""),
        )
    )

    return record_tariff_proof(
        law_context=law_context,
        base_facts=dict(baseline_facts),
        mutations=lever.mutation if lever else {},
        baseline_active=baseline_active,
        optimized_active=optimized_active,
        baseline_sat=baseline_eval.sat,
        optimized_sat=optimized_eval.sat,
        baseline_duty_rate=baseline_rate,
        optimized_duty_rate=optimized_rate,
        baseline_duty_status=baseline_eval.duty_status,
        optimized_duty_status=optimized_eval.duty_status,
        baseline_scenario=dict(baseline_facts),
        optimized_scenario=dict(optimized_facts),
        baseline_unsat_core=baseline_unsat,
        optimized_unsat_core=optimized_unsat,
        provenance_chain=provenance_chain,
        evidence_pack=evidence_pack,
        overlays=overlays,
        tariff_manifest_hash=context_meta.get("manifest_hash"),
    )


def suggest_tariff_optimizations(
    request: TariffSuggestRequestModel,
) -> TariffSuggestResponseModel:
    """Generate, evaluate, and rank tariff optimization suggestions."""

    bom_structured = request.bom_json
    if bom_structured is None and request.bom_csv:
        bom_structured = parse_bom_csv(request.bom_csv)

    normalized_bom_text = request.bom_text
    if bom_structured is not None:
        normalized_bom_text = bom_to_text(bom_structured)

    profile = infer_product_profile(request.description, normalized_bom_text)
    spec, extraction_evidence = extract_product_spec(
        request.description,
        normalized_bom_text,
        bom_structured=bom_structured,
        origin_country=request.origin_country,
        export_country=request.export_country,
        import_country=request.import_country,
    )
    compiled_facts, fact_evidence = compile_facts_with_evidence(
        spec,
        request.description,
        normalized_bom_text,
        bom_structured=bom_structured,
        include_fact_evidence=request.include_fact_evidence,
    )
    if fact_evidence:
        fact_evidence = sorted(
            fact_evidence,
            key=lambda item: (
                item.fact_key,
                json.dumps(item.value, sort_keys=True),
                len(item.evidence),
            ),
        )
    if extraction_evidence:
        extraction_evidence = sorted(
            extraction_evidence,
            key=lambda item: (item.source, item.start or -1, item.end or -1, item.snippet),
        )

    baseline_facts = baseline_facts_from_profile(profile)
    if compiled_facts:
        for key, value in compiled_facts.items():
            if value or key not in baseline_facts:
                baseline_facts[key] = value

    normalized_facts, normalization_notes = normalize_compiled_facts(baseline_facts)

    material_parts = [mat.material for mat in spec.materials]
    if bom_structured is not None:
        material_parts.extend(item.material or "" for item in bom_structured.items if item.material)
    material_text = " ".join(part for part in material_parts if part)
    intended_use = spec.use_function or ""
    declared_hts = str(normalized_facts.get("hts_code") or "").strip() or None
    if declared_hts is None and bom_structured is not None:
        for item in bom_structured.items:
            if item.hts_code:
                declared_hts = item.hts_code
                normalized_facts["hts_code"] = declared_hts
                break
    auto_classification_payload = build_auto_classification_payload(
        description=request.description,
        material=material_text,
        intended_use=intended_use,
        declared_hts=declared_hts,
    )
    auto_classification = AutoClassificationResultModel(**auto_classification_payload)

    if auto_classification.hts_source == "auto_classified" and auto_classification.selected_hts_code:
        normalized_facts["hts_code"] = auto_classification.selected_hts_code

    missing_inputs = validate_minimum_inputs(spec.model_dump(), normalized_facts)
    compiled_pack = {"raw": compiled_facts, "normalized": normalized_facts}
    compiled_pack.update(compiled_facts)
    compiled_pack["auto_classification"] = auto_classification.model_dump()
    compiled_pack["hts_source"] = auto_classification.hts_source
    if auto_classification.mismatch_flag and auto_classification.review_reason:
        normalization_notes.append(auto_classification.review_reason)

    resolved_context = request.law_context or DEFAULT_CONTEXT_ID
    atoms, context_meta = load_atoms_for_context(resolved_context)
    law_context = context_meta["context_id"]
    duty_rates = context_meta.get("duty_rates") or DUTY_RATES

    declared_value = request.declared_value_per_unit or 100.0
    seed = request.seed or 2025  # reserved for future stochastic flows

    overlay_context = {
        "origin_country": request.origin_country,
        "import_country": request.import_country,
        "hts_code": normalized_facts.get("hts_code"),
    }
    baseline_candidates = generate_baseline_candidates(normalized_facts, atoms, duty_rates, max_candidates=5)
    baseline_eval = _evaluate_scenario(
        atoms, normalized_facts, duty_rates, overlay_context=overlay_context
    )
    baseline_safe_rate, baseline_safe_confidence = _select_baseline_rate(baseline_eval, baseline_candidates, duty_rates)
    baseline_rate_raw = (
        baseline_safe_rate
        if baseline_safe_rate is not None
        else (baseline_candidates[0].duty_rate if baseline_candidates else baseline_eval.duty_rate)
    )
    baseline_rate = baseline_safe_rate if baseline_safe_rate is not None else baseline_eval.effective_duty_rate
    baseline_confidence = (
        baseline_safe_confidence
        if baseline_safe_confidence is not None
        else (baseline_candidates[0].confidence if baseline_candidates else 0.3)
    )

    suggestion_items: List[TariffSuggestionItemModel] = []
    why_not_optimized: List[str] = normalization_notes[:]
    proof_id: str | None = None
    proof_payload_hash: str | None = None
    stop_overlays = [ov for ov in baseline_eval.overlays if ov.stop_optimization]

    if missing_inputs:
        return TariffSuggestResponseModel(
            status="INSUFFICIENT_INPUTS",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=0,
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            auto_classification=auto_classification,
            why_not_optimized=missing_inputs + why_not_optimized,
            proof_id=None,
            proof_payload_hash=None,
        )

    evidence_pack = {
        "product_spec": spec.model_dump(),
        "compiled_facts": compiled_pack,
        "fact_evidence": [item.model_dump() for item in fact_evidence] if fact_evidence else [],
        "extraction_evidence": [item.model_dump() for item in extraction_evidence] if extraction_evidence else [],
    }
    sku_metadata = build_sku_metadata_snapshot(
        sku_id=request.sku_id,
        description=request.description,
        bom_json=request.bom_json,
        bom_csv=request.bom_csv,
        origin_country=request.origin_country,
        export_country=request.export_country,
        import_country=request.import_country,
        declared_value_per_unit=request.declared_value_per_unit,
        annual_volume=request.annual_volume,
    )
    if sku_metadata:
        evidence_pack["sku_metadata"] = sku_metadata

    if not baseline_eval.sat:
        return TariffSuggestResponseModel(
            status="INSUFFICIENT_RULE_COVERAGE",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=0,
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            auto_classification=auto_classification,
            why_not_optimized=why_not_optimized,
            proof_id=None,
            proof_payload_hash=None,
        )

    if stop_overlays:
        review_reasons = [
            f"{ov.overlay_name}: {ov.reason}" for ov in sorted(stop_overlays, key=lambda item: item.overlay_name)
        ]
        baseline_proof = _record_proof(
            law_context=law_context,
            context_meta=context_meta,
            baseline_facts=normalized_facts,
            optimized_facts=normalized_facts,
            baseline_eval=baseline_eval,
            optimized_eval=baseline_eval,
            lever=None,
            evidence_pack=evidence_pack,
            overlays={"baseline": [ov.model_dump() for ov in baseline_eval.overlays], "optimized": []},
        )
        return TariffSuggestResponseModel(
            status="REQUIRES_REVIEW",
            sku_id=request.sku_id,
            description=request.description,
            law_context=law_context,
            baseline_scenario=normalized_facts,
            generated_candidates_count=0,
            suggestions=[],
            tariff_manifest_hash=context_meta["manifest_hash"],
            fact_evidence=fact_evidence if request.include_fact_evidence else None,
            product_spec=spec if request.include_fact_evidence else None,
            baseline_candidates=baseline_candidates,
            auto_classification=auto_classification,
            why_not_optimized=review_reasons + why_not_optimized,
            proof_id=baseline_proof.proof_id,
            proof_payload_hash=baseline_proof.proof_payload_hash,
        )

    dynamic_levers: list[LeverModel] = []
    if baseline_eval.duty_atoms:
        dynamic_levers = generate_candidate_levers(
            baseline_atom=baseline_eval.duty_atoms[0],
            atoms=atoms,
            duty_rates=duty_rates,
            facts=normalized_facts,
            baseline_rate=baseline_rate,
            fact_evidence=fact_evidence,
            declared_value_per_unit=declared_value,
            annual_volume=request.annual_volume,
            min_net_savings=request.min_net_savings,
            max_payback_months=request.max_payback_months,
            risk_tolerance=request.risk_tolerance,
            stop_optimization=any(ov.stop_optimization for ov in baseline_eval.overlays),
        )

    lever_index: dict[str, LeverModel] = {
        lever.lever_id: lever for lever in applicable_levers(spec=spec, facts=normalized_facts)
    }
    for lever in dynamic_levers:
        lever_index.setdefault(lever.lever_id, lever)
    levers = sorted(lever_index.values(), key=lambda lever: lever.lever_id)

    for lever in levers:
        mutated = deepcopy(normalized_facts)
        mutated.update(lever.mutation)
        mutated_normalized, _ = normalize_compiled_facts(mutated)

        optimized_eval = _evaluate_scenario(
            atoms, mutated_normalized, duty_rates, overlay_context=overlay_context
        )
        optimized_rate = optimized_eval.effective_duty_rate
        optimized_duty_atoms = optimized_eval.duty_atoms
        if not optimized_eval.sat or optimized_rate is None:
            continue
        if optimized_eval.duty_rate is None and not optimized_eval.overlays:
            continue
        if any(ov.stop_optimization for ov in optimized_eval.overlays):
            why_not_optimized.append(
                "overlay gating optimization: "
                + "; ".join(f"{ov.overlay_name}: {ov.reason}" for ov in optimized_eval.overlays if ov.stop_optimization)
            )
            continue

        optimized_candidates = generate_baseline_candidates(mutated_normalized, atoms, duty_rates, max_candidates=1)
        optimized_confidence = optimized_candidates[0].confidence if optimized_candidates else baseline_confidence

        savings_rate, savings_value, annual_value = _compute_savings(
            baseline_rate=baseline_rate,
            optimized_rate=optimized_rate,
            declared_value_per_unit=declared_value,
            annual_volume=request.annual_volume,
        )
        feasibility_profile = _feasibility_for_lever(lever)
        net_savings = _compute_net_savings(
            baseline_rate=baseline_rate,
            optimized_rate=optimized_rate,
            declared_value_per_unit=declared_value,
            annual_volume=request.annual_volume,
            feasibility=feasibility_profile,
        )
        ranking_score = (
            net_savings.net_annual_savings
            if net_savings.net_annual_savings is not None
            else (annual_value if annual_value is not None else savings_value)
        )

        confidence_gain = optimized_confidence > baseline_confidence + 0.2
        if savings_rate <= 0 and not confidence_gain:
            continue

        baseline_active_atoms = baseline_eval.active_atoms
        baseline_unsat = baseline_eval.unsat_core
        baseline_duty_atoms = baseline_eval.duty_atoms
        risk = assess_tariff_risk(
            baseline_facts=normalized_facts,
            optimized_facts=mutated_normalized,
            baseline_active_atoms=baseline_active_atoms,
            optimized_active_atoms=optimized_eval.active_atoms,
            baseline_duty_rate=baseline_rate or optimized_rate,
            optimized_duty_rate=optimized_rate,
        )
        adjusted_risk_score = max(risk.risk_score, lever.risk_floor)
        adjusted_grade = _defensibility_grade(adjusted_risk_score)

        proof_handle = _record_proof(
            law_context=law_context,
            context_meta=context_meta,
            baseline_facts=normalized_facts,
            optimized_facts=mutated_normalized,
            baseline_eval=baseline_eval,
            optimized_eval=optimized_eval,
            lever=lever,
            evidence_pack=evidence_pack,
            overlays={
                "baseline": [ov.model_dump() for ov in baseline_eval.overlays],
                "optimized": [ov.model_dump() for ov in optimized_eval.overlays],
            },
        )

        provenance_chain: List[Dict[str, Any]] = []
        provenance_chain.extend(_build_provenance(baseline_duty_atoms, "baseline"))
        provenance_chain.extend(_build_provenance(optimized_duty_atoms, "optimized"))
        provenance_chain.sort(
            key=lambda item: (
                item.get("source_id", ""),
                item.get("section", ""),
                item.get("text", ""),
                item.get("scenario", ""),
            )
        )

        documentation_fields = {}
        if lever.lever_type == "DOCUMENTATION":
            why_needed = list(lever.why_needed) + list(lever.measurement_hints)
            documentation_fields = {
                "lever_category": DOCUMENTATION_LEVER_ID,
                "accepted_evidence_templates": list(lever.evidence_requirements),
                "fact_gaps": list(lever.fact_gaps),
                "why_needed": sorted(set(why_needed)),
            }

        suggestion_items.append(
            TariffSuggestionItemModel(
                human_summary=lever.rationale,
                lever_id=lever.lever_id,
                lever_feasibility=lever.feasibility,
                feasibility=feasibility_profile,
                evidence_requirements=list(lever.evidence_requirements),
                baseline_duty_rate=baseline_rate_raw or baseline_eval.duty_rate or 0.0,
                optimized_duty_rate=optimized_eval.duty_rate if optimized_eval.duty_rate is not None else optimized_rate,
                baseline_effective_duty_rate=baseline_rate,
                optimized_effective_duty_rate=optimized_rate,
                savings_per_unit_rate=savings_rate,
                savings_per_unit_value=savings_value,
                annual_savings_value=annual_value,
                net_savings=net_savings,
                ranking_score=ranking_score,
                best_mutation=dict(lever.mutation),
                classification_confidence=optimized_confidence,
                active_codes_baseline=sorted({atom.source_id for atom in baseline_duty_atoms}),
                active_codes_optimized=sorted({atom.source_id for atom in optimized_duty_atoms}),
                provenance_chain=provenance_chain,
                overlays={"baseline": baseline_eval.overlays, "optimized": optimized_eval.overlays},
                law_context=law_context,
                proof_id=proof_handle.proof_id,
                proof_payload_hash=proof_handle.proof_payload_hash,
                risk_score=adjusted_risk_score,
                defensibility_grade=adjusted_grade,
                risk_reasons=risk.risk_reasons,
                tariff_manifest_hash=context_meta["manifest_hash"],
                strategy_type=_strategy_type_for_lever(lever),
                risk_level=_risk_level_for_score(adjusted_risk_score),
                required_actions=["Validate operational feasibility and execute mutation with broker review."],
                documentation_required=list(lever.evidence_requirements),
                confidence_level=optimized_confidence,
                implementation_difficulty=(lever.feasibility or "medium").lower(),
                **documentation_fields,
            )
        )

    documentation_levers = _documentation_levers(
        baseline_candidates=baseline_candidates,
        atoms=atoms,
        duty_rates=duty_rates,
        baseline_eval=baseline_eval,
        baseline_facts=normalized_facts,
        baseline_rate=baseline_rate,
        baseline_rate_raw=baseline_rate_raw,
        baseline_confidence=baseline_confidence,
        declared_value=declared_value,
        annual_volume=request.annual_volume,
        law_context=law_context,
        context_meta=context_meta,
        evidence_pack=evidence_pack,
        overlay_context=overlay_context,
    )
    suggestion_items.extend(documentation_levers)

    baseline_proof_for_advisories = _record_proof(
        law_context=law_context,
        context_meta=context_meta,
        baseline_facts=normalized_facts,
        optimized_facts=normalized_facts,
        baseline_eval=baseline_eval,
        optimized_eval=baseline_eval,
        lever=None,
        evidence_pack=evidence_pack,
        overlays={
            "baseline": [ov.model_dump() for ov in baseline_eval.overlays],
            "optimized": [ov.model_dump() for ov in baseline_eval.overlays],
        },
    )

    has_declared_classification = bool(normalized_facts.get("hts_code"))
    reclass_candidates: list[dict[str, Any]] = []
    if has_declared_classification:
        reclass_candidates = build_reclassification_candidates(
            current_hts=str(normalized_facts.get("hts_code") or ""),
            baseline_rate=baseline_rate,
            description=request.description,
            material=material_text,
            annual_volume=request.annual_volume,
            declared_value_per_unit=declared_value,
            material_breakdown=[item.model_dump() for item in spec.materials],
        )
    if has_declared_classification and (not reclass_candidates) and auto_classification.alternatives and baseline_rate is not None:
        current_hts = str(normalized_facts.get("hts_code") or "")
        for alternative in auto_classification.alternatives:
            if not alternative.hts_code or alternative.hts_code == current_hts:
                continue
            if alternative.duty_rate >= baseline_rate:
                continue
            savings_rate = baseline_rate - alternative.duty_rate
            savings_value = savings_rate / 100.0 * declared_value
            annual_savings = savings_value * request.annual_volume if request.annual_volume is not None else None
            reclass_candidates.append(
                {
                    "strategy_type": "reclassification",
                    "from_hts": current_hts,
                    "to_hts": alternative.hts_code,
                    "candidate_description": alternative.description,
                    "optimized_duty_rate": alternative.duty_rate,
                    "savings_per_unit_rate": savings_rate,
                    "savings_per_unit_value": savings_value,
                    "annual_savings_value": annual_savings,
                    "plausibility_score": alternative.confidence,
                    "confidence_level": alternative.confidence,
                    "risk_level": "medium_risk",
                    "risk_rationale": "Alternative heading inferred from text; broker review required.",
                    "required_actions": [
                        "Validate candidate heading language against BOM and intended use.",
                        "Prepare backup ruling request package.",
                    ],
                    "documentation_required": ["BOM and datasheet", "Product photos and drawings"],
                    "implementation_difficulty": "medium",
                }
            )
            break
    risk_to_score = {"low_risk": 20, "medium_risk": 45, "high_risk": 75}

    for candidate in reclass_candidates:
        risk_level = str(candidate["risk_level"])
        risk_score = risk_to_score.get(risk_level, 70)
        suggestion_items.append(
            TariffSuggestionItemModel(
                human_summary=(
                    f"Reclassify from {candidate['from_hts']} to {candidate['to_hts']} when product details match "
                    "the lower-duty heading."
                ),
                lever_id=f"RECLASS_{str(candidate['to_hts']).replace('.', '_')}",
                optimization_type="RECLASSIFICATION",
                strategy_type="reclassification",
                lever_feasibility=str(candidate.get("implementation_difficulty", "medium")).upper(),
                evidence_requirements=list(candidate.get("documentation_required", [])),
                baseline_duty_rate=baseline_rate_raw or baseline_eval.duty_rate or 0.0,
                optimized_duty_rate=float(candidate["optimized_duty_rate"]),
                baseline_effective_duty_rate=baseline_rate,
                optimized_effective_duty_rate=float(candidate["optimized_duty_rate"]),
                savings_per_unit_rate=float(candidate["savings_per_unit_rate"]),
                savings_per_unit_value=float(candidate["savings_per_unit_value"]),
                annual_savings_value=candidate.get("annual_savings_value"),
                ranking_score=float(
                    candidate.get("annual_savings_value")
                    if candidate.get("annual_savings_value") is not None
                    else candidate["savings_per_unit_value"]
                ),
                best_mutation={"hts_code": candidate["to_hts"]},
                classification_confidence=float(candidate.get("plausibility_score") or 0.0),
                active_codes_baseline=sorted({atom.source_id for atom in baseline_eval.duty_atoms}),
                active_codes_optimized=[str(candidate["to_hts"])],
                provenance_chain=[],
                overlays={"baseline": baseline_eval.overlays, "optimized": baseline_eval.overlays},
                law_context=law_context,
                proof_id=baseline_proof_for_advisories.proof_id,
                proof_payload_hash=baseline_proof_for_advisories.proof_payload_hash,
                risk_score=risk_score,
                defensibility_grade=_defensibility_grade(risk_score),
                risk_reasons=[str(candidate.get("risk_rationale") or "")],
                tariff_manifest_hash=context_meta["manifest_hash"],
                risk_level=risk_level,
                required_actions=list(candidate.get("required_actions") or []),
                documentation_required=list(candidate.get("documentation_required") or []),
                confidence_level=float(candidate.get("confidence_level") or 0.0),
                implementation_difficulty=str(candidate.get("implementation_difficulty") or "medium"),
            )
        )

    advisory_items: list[dict[str, Any]] = []
    if has_declared_classification or bom_structured is not None:
        advisory_items = build_advisory_strategies(
            origin_country=request.origin_country,
            bom_items=[item.model_dump() for item in bom_structured.items] if bom_structured is not None else [],
            baseline_rate=baseline_rate,
            declared_value_per_unit=declared_value,
            annual_volume=request.annual_volume,
            material_breakdown=[item.model_dump() for item in spec.materials],
        )
    for advisory in advisory_items:
        risk_level = str(advisory.get("risk_level") or "medium_risk")
        risk_score = risk_to_score.get(risk_level, 50)
        suggestion_items.append(
            TariffSuggestionItemModel(
                human_summary=str(advisory["human_summary"]),
                lever_id=f"ADVISORY_{str(advisory['strategy_type']).upper()}",
                optimization_type="ADVISORY",
                strategy_type=str(advisory["strategy_type"]),
                lever_feasibility=str(advisory.get("implementation_difficulty", "medium")).upper(),
                evidence_requirements=list(advisory.get("documentation_required", [])),
                baseline_duty_rate=baseline_rate_raw or baseline_eval.duty_rate or 0.0,
                optimized_duty_rate=float(advisory.get("optimized_duty_rate") or (baseline_rate or 0.0)),
                baseline_effective_duty_rate=baseline_rate,
                optimized_effective_duty_rate=float(advisory.get("optimized_duty_rate") or (baseline_rate or 0.0)),
                savings_per_unit_rate=float(advisory.get("savings_per_unit_rate") or 0.0),
                savings_per_unit_value=float(advisory.get("savings_per_unit_value") or 0.0),
                annual_savings_value=advisory.get("annual_savings_value"),
                ranking_score=float(
                    advisory.get("annual_savings_value")
                    if advisory.get("annual_savings_value") is not None
                    else advisory.get("savings_per_unit_value") or 0.0
                ),
                best_mutation={},
                classification_confidence=float(advisory.get("confidence_level") or 0.0),
                active_codes_baseline=sorted({atom.source_id for atom in baseline_eval.duty_atoms}),
                active_codes_optimized=sorted({atom.source_id for atom in baseline_eval.duty_atoms}),
                provenance_chain=[],
                overlays={"baseline": baseline_eval.overlays, "optimized": baseline_eval.overlays},
                law_context=law_context,
                proof_id=baseline_proof_for_advisories.proof_id,
                proof_payload_hash=baseline_proof_for_advisories.proof_payload_hash,
                risk_score=risk_score,
                defensibility_grade=_defensibility_grade(risk_score),
                risk_reasons=[str(advisory.get("risk_rationale") or "")],
                tariff_manifest_hash=context_meta["manifest_hash"],
                risk_level=risk_level,
                required_actions=list(advisory.get("required_actions") or []),
                documentation_required=list(advisory.get("documentation_required") or []),
                confidence_level=float(advisory.get("confidence_level") or 0.0),
                implementation_difficulty=str(advisory.get("implementation_difficulty") or "medium"),
            )
        )

    atoms_by_id = {atom.source_id: atom for atom in atoms}
    suggestion_items = [_apply_precedent_signals(item, atoms_by_id) for item in suggestion_items]

    suggestion_items = [item for item in suggestion_items if _passes_constraints(item, request)]

    indexed_items = list(enumerate(suggestion_items))
    indexed_items.sort(key=lambda pair: _sort_key(pair[1], request.annual_volume, pair[0]))
    top_k = request.top_k or 5
    ordered_suggestions = [item for _, item in indexed_items[:top_k]]

    status = "OK_OPTIMIZED" if ordered_suggestions else "OK_BASELINE_ONLY"
    if not ordered_suggestions:
        why_not_optimized.append("no feasible optimization identified")
        baseline_proof = _record_proof(
            law_context=law_context,
            context_meta=context_meta,
            baseline_facts=normalized_facts,
            optimized_facts=normalized_facts,
            baseline_eval=baseline_eval,
            optimized_eval=baseline_eval,
            lever=None,
            evidence_pack=evidence_pack,
            overlays={
                "baseline": [ov.model_dump() for ov in baseline_eval.overlays],
                "optimized": [ov.model_dump() for ov in baseline_eval.overlays],
            },
        )
        proof_id = baseline_proof.proof_id
        proof_payload_hash = baseline_proof.proof_payload_hash
    else:
        proof_id = ordered_suggestions[0].proof_id
        proof_payload_hash = ordered_suggestions[0].proof_payload_hash

    return TariffSuggestResponseModel(
        status=status,
        sku_id=request.sku_id,
        description=request.description,
        law_context=law_context,
        baseline_scenario=normalized_facts,
        generated_candidates_count=len(levers),
        suggestions=ordered_suggestions,
        tariff_manifest_hash=context_meta["manifest_hash"],
        fact_evidence=fact_evidence if request.include_fact_evidence else None,
        product_spec=spec if request.include_fact_evidence else None,
        baseline_candidates=baseline_candidates,
        auto_classification=auto_classification,
        why_not_optimized=why_not_optimized,
        proof_id=proof_id,
        proof_payload_hash=proof_payload_hash,
    )
