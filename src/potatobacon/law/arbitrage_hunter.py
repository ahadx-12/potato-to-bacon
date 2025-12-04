"""Arbitrage hunter built on top of the CALE-LAW pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set

from z3 import BoolVal, If, Optimize, sat  # type: ignore[import-not-found]

from potatobacon.cale.bootstrap import CALEServices
from potatobacon.law.cale_metrics import ScenarioMetrics, compute_scenario_metrics, sample_scenarios
from potatobacon.law.arbitrage_models import (
    ArbitrageCandidateModel,
    ArbitrageDossierModel,
    ArbitrageMetrics,
    ArbitrageScenario,
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    ProvenanceStep,
)
from potatobacon.law.solver_z3 import PolicyAtom, build_policy_atoms_from_rules, compile_atoms_to_z3


@dataclass(slots=True)
class ArbitrageRequest:
    jurisdictions: List[str]
    domain: str
    objective: str
    constraints: Dict[str, Any]
    risk_tolerance: str
    seed: int | None = None


@dataclass(slots=True)
class ArbitrageCandidate:
    scenario: Dict[str, bool]
    metrics: ScenarioMetrics
    proof_trace: List[str]


@dataclass(slots=True)
class ArbitrageDossier:
    golden_scenario: Dict[str, bool]
    metrics: Dict[str, float]
    proof_trace: List[str]
    risk_flags: List[str]
    candidates: List[ArbitrageCandidate]
    provenance_chain: List[ProvenanceStep]
    dependency_graph: DependencyGraph | None


class ArbitrageHunter:
    """Explore the CALE rule space for high-value ambiguous scenarios."""

    def __init__(self, atoms: Sequence[PolicyAtom], seed: int | None = None):
        self._atoms = list(atoms)
        self._seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    @classmethod
    def from_rules(cls, rules: Sequence[Any]) -> "ArbitrageHunter":
        atoms = build_policy_atoms_from_rules(rules, mapper=None)
        return cls(atoms)

    def _filter_atoms(self, jurisdictions: Sequence[str]) -> List[PolicyAtom]:
        if not jurisdictions:
            return list(self._atoms)
        jurisdictions_lower = {j.lower() for j in jurisdictions}
        return [
            atom
            for atom in self._atoms
            if atom.outcome.get("jurisdiction", "").lower() in jurisdictions_lower
        ]

    def _optimise_seed(self, atoms: Sequence[PolicyAtom], constraints: Mapping[str, Any]) -> Dict[str, bool]:
        optimizer = Optimize()
        var_map = compile_atoms_to_z3(atoms, {})

        # Apply user constraints
        for key, value in constraints.items():
            if key not in var_map:
                continue
            optimizer.add(var_map[key] == BoolVal(bool(value)))

        # Encourage satisfying as many outcomes as possible to expose ambiguity
        objective_terms = []
        for atom in atoms:
            if atom.z3_guard is None or atom.z3_outcome is None:
                continue
            objective_terms.append(If(atom.z3_guard, If(atom.z3_outcome, 1, 1), 0))
        if objective_terms:
            optimizer.maximize(sum(objective_terms))

        result = optimizer.check()
        if result != sat:
            return {}
        model = optimizer.model()
        scenario: Dict[str, bool] = {}
        for name, var in var_map.items():
            value = model.eval(var, model_completion=True)
            scenario[name] = bool(value)
        return scenario

    def _mutate(self, scenario: Dict[str, bool], mutations: int = 2) -> Dict[str, bool]:
        mutated = dict(scenario)
        keys = list(mutated.keys())
        for _ in range(max(1, mutations)):
            key = self._rng.choice(keys)
            mutated[key] = not mutated[key]
        return mutated

    @staticmethod
    def _role_from_modality(modality: str) -> str:
        modality_upper = modality.upper()
        if modality_upper == "OBLIGE":
            return "obligation"
        if modality_upper == "FORBID":
            return "prohibition"
        if modality_upper == "PERMIT":
            return "permission"
        return "classification"

    def _provenance_chain(
        self, active_ids: Set[str], request_jurisdictions: Sequence[str], atoms: Sequence[PolicyAtom]
    ) -> List[ProvenanceStep]:
        jurisdictions_lower = {j.lower() for j in request_jurisdictions}
        chain: List[ProvenanceStep] = []
        seen: Set[str] = set()
        for atom in atoms:
            jurisdiction = atom.outcome.get("jurisdiction", "")
            if jurisdictions_lower and jurisdiction.lower() not in jurisdictions_lower:
                continue
            if atom.source_id in seen:
                continue
            if active_ids and atom.source_id not in active_ids:
                continue
            urn = self._build_urn(atom)
            citations = self._citations(atom)
            seen.add(atom.source_id)
            chain.append(
                ProvenanceStep(
                    step=len(chain) + 1,
                    jurisdiction=jurisdiction or (request_jurisdictions[0] if request_jurisdictions else "Unknown"),
                    rule_id=atom.source_id,
                    type=atom.rule_type or "RULE",
                    role=self._role_from_modality(atom.modality or atom.outcome.get("modality", "")),
                    summary=(atom.text or atom.action or atom.source_id)[:120],
                    atom_id=atom.atom_id,
                    urn=urn,
                    citations=citations,
                    effective_date=self._effective_date(atom),
                )
            )
        if len(chain) < 2:
            for atom in atoms:
                if atom.source_id in seen:
                    continue
                if jurisdictions_lower and atom.outcome.get("jurisdiction", "").lower() not in jurisdictions_lower:
                    continue
                urn = self._build_urn(atom)
                citations = self._citations(atom)
                chain.append(
                    ProvenanceStep(
                        step=len(chain) + 1,
                        jurisdiction=atom.outcome.get("jurisdiction")
                        or (request_jurisdictions[0] if request_jurisdictions else "Unknown"),
                        rule_id=atom.source_id,
                        type=atom.rule_type or "RULE",
                        role=self._role_from_modality(atom.modality or atom.outcome.get("modality", "")),
                        summary=(atom.text or atom.action or atom.source_id)[:120],
                        atom_id=atom.atom_id,
                        urn=urn,
                        citations=citations,
                        effective_date=self._effective_date(atom),
                    )
                )
                if len(chain) >= 2:
                    break
        return chain

    def _atom_label(self, rule_id: str, atoms: Sequence[PolicyAtom]) -> str:
        for atom in atoms:
            if atom.source_id == rule_id:
                label_parts = [atom.statute or atom.source_id, atom.section]
                return " ".join(part for part in label_parts if part).strip() or atom.source_id
        return rule_id

    def _dependency_graph(
        self, provenance: List[ProvenanceStep], atoms: Sequence[PolicyAtom]
    ) -> DependencyGraph | None:
        if not provenance:
            return None
        nodes: Dict[str, DependencyNode] = {}
        for step in provenance:
            if step.rule_id not in nodes:
                nodes[step.rule_id] = DependencyNode(
                    id=step.rule_id,
                    jurisdiction=step.jurisdiction,
                    label=self._atom_label(step.rule_id, atoms),
                    urn=step.urn,
                    citations=step.citations,
                )
        edges: List[DependencyEdge] = []
        for current, nxt in zip(provenance, provenance[1:]):
            edges.append(DependencyEdge(from_id=current.rule_id, to_id=nxt.rule_id, relation="sequence"))
        return DependencyGraph(nodes=list(nodes.values()), edges=edges)

    @staticmethod
    def _build_urn(atom: PolicyAtom) -> str:
        jurisdiction = atom.outcome.get("jurisdiction", "unknown") or "unknown"
        statute = atom.statute or atom.source_id
        section = atom.section or "0"
        jurisdiction_token = jurisdiction.lower().replace(" ", "_").replace(".", "_")
        statute_token = statute.lower().replace(" ", "_").replace(".", "_")
        section_token = section.lower().replace(" ", "_").replace(".", "_")
        return f"urn:law:{jurisdiction_token}:{statute_token}:s{section_token}"

    @staticmethod
    def _citations(atom: PolicyAtom) -> List[str]:
        citation = " ".join(part for part in [atom.statute, atom.section] if part).strip()
        if not citation:
            citation = atom.source_id
        return [citation]

    @staticmethod
    def _effective_date(atom: PolicyAtom) -> str | None:
        enactment = getattr(atom, "enactment_year", None)
        if enactment:
            return str(enactment)
        return None

    def hunt(self, request: ArbitrageRequest) -> ArbitrageDossier:
        atoms = self._filter_atoms(request.jurisdictions)
        if not atoms:
            atoms = list(self._atoms)

        base_constraints = request.constraints or {}
        seed_scenario = self._optimise_seed(atoms, base_constraints)
        if not seed_scenario:
            seed_candidates = sample_scenarios(atoms, sample_size=5, rng=self._rng)
        else:
            seed_candidates = [seed_scenario]

        fuzz_budget = 5 if request.risk_tolerance == "low" else 10
        candidates: List[ArbitrageCandidate] = []
        for seed in seed_candidates:
            metrics = compute_scenario_metrics(seed, atoms, seed=self._seed)
            proof = [atom.outcome_label for atom in atoms if atom.source_id in metrics.active_rules]
            if not proof and atoms:
                proof = [atoms[0].outcome_label]
            candidates.append(ArbitrageCandidate(seed, metrics, proof))
            for _ in range(fuzz_budget):
                mutated = self._mutate(seed)
                metrics_mut = compute_scenario_metrics(mutated, atoms, seed=self._seed)
                proof_mut = [
                    atom.outcome_label for atom in atoms if atom.source_id in metrics_mut.active_rules
                ]
                if not proof_mut and atoms:
                    proof_mut = [atoms[0].outcome_label]
                candidates.append(ArbitrageCandidate(mutated, metrics_mut, proof_mut))

        top_candidates = sorted(candidates, key=lambda c: c.metrics.score, reverse=True)[:5]
        if top_candidates:
            golden = top_candidates[0]
        else:
            golden = ArbitrageCandidate({}, compute_scenario_metrics({}, atoms, seed=self._seed), [])

        jurisdiction_span = max(1, len(set(request.jurisdictions)))
        value_boost = 1.0 + 0.05 * (jurisdiction_span - 1)
        adjusted_value = golden.metrics.value_estimate * value_boost
        adjusted_risk = max(0.0, golden.metrics.risk - 0.02 * (jurisdiction_span - 1))

        dossier_metrics = ArbitrageMetrics(
            value=adjusted_value,
            entropy=golden.metrics.entropy,
            kappa=golden.metrics.kappa,
            risk=adjusted_risk,
            contradiction_probability=1.0 if golden.metrics.contradiction else 0.0,
            score=golden.metrics.score * value_boost,
            value_components=golden.metrics.value_components,
            risk_components=golden.metrics.risk_components,
            score_components=golden.metrics.score_components,
            alpha=golden.metrics.alpha,
            beta=golden.metrics.beta,
            seed=golden.metrics.seed,
        )
        risk_flags = []
        if golden.metrics.risk > 0.6:
            risk_flags.append("High ambiguity relative to dominant outcome")
        if golden.metrics.contradiction:
            risk_flags.append("Scenario leads to inconsistent obligations")

        active_ids = set(golden.metrics.active_rules)
        provenance_chain = self._provenance_chain(active_ids, request.jurisdictions, atoms)
        dependency_graph = self._dependency_graph(provenance_chain, atoms)
        golden_scenario = ArbitrageScenario(jurisdictions=list(request.jurisdictions), facts=golden.scenario)

        return ArbitrageDossier(
            golden_scenario=golden_scenario.model_dump(),
            metrics=dossier_metrics.model_dump(),
            proof_trace=golden.proof_trace,
            risk_flags=risk_flags,
            candidates=top_candidates,
            provenance_chain=provenance_chain,
            dependency_graph=dependency_graph,
        )


def run_arbitrage_hunt(services: CALEServices, req: Mapping[str, Any]) -> Dict[str, Any]:
    atoms = build_policy_atoms_from_rules(services.corpus, services.mapper)
    hunter = ArbitrageHunter(atoms, seed=req.get("seed"))
    dossier = hunter.hunt(
        ArbitrageRequest(
            jurisdictions=list(req.get("jurisdictions", [])),
            domain=req.get("domain", "tax"),
            objective=req.get("objective", ""),
            constraints=req.get("constraints", {}),
            risk_tolerance=req.get("risk_tolerance", "medium"),
        )
    )

    response_candidates: List[ArbitrageCandidateModel] = []
    for candidate in dossier.candidates:
        metrics = candidate.metrics
        metrics_model = ArbitrageMetrics(
            value=metrics.value_estimate,
            entropy=metrics.entropy,
            kappa=metrics.kappa,
            risk=metrics.risk,
            contradiction_probability=1.0 if metrics.contradiction else 0.0,
            score=metrics.score,
            value_components=metrics.value_components,
            risk_components=metrics.risk_components,
            score_components=metrics.score_components,
            alpha=metrics.alpha,
            beta=metrics.beta,
            seed=metrics.seed,
        )
        response_candidates.append(
            ArbitrageCandidateModel(
                scenario=candidate.scenario,
                metrics=metrics_model,
                proof_trace=candidate.proof_trace,
            )
        )

    dossier_model = ArbitrageDossierModel(
        golden_scenario=ArbitrageScenario(**dossier.golden_scenario),
        metrics=ArbitrageMetrics(**dossier.metrics),
        proof_trace=dossier.proof_trace,
        risk_flags=dossier.risk_flags,
        candidates=response_candidates,
        provenance_chain=dossier.provenance_chain,
        dependency_graph=dossier.dependency_graph,
    )

    return dossier_model.model_dump()
