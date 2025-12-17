"""Z3 helpers for translating CALE policy atoms into constraints.

This module purposely reuses the existing CALE rule representation so that the
legal analytics endpoints can reason about ambiguity and contradictions without
introducing a new DSL.  The helpers here are intentionally lightweight: they
compile the familiar ``LegalRule`` objects into :class:`z3.BoolRef` guards and
outcomes that can be composed by downstream search/metrics components.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from z3 import (  # type: ignore[import-not-found]
    And,
    Bool,
    BoolRef,
    BoolVal,
    Implies,
    Not,
    Optimize,
    Solver,
    sat,
    unsat,
)

from potatobacon.cale.parser import PredicateMapper
from potatobacon.cale.types import LegalRule

_ATOM_CACHE: Dict[Tuple[str, Tuple[str, ...]], Dict[str, object]] = {}
_Z3_LOCK = threading.Lock()


@dataclass(slots=True)
class PolicyAtom:
    """Minimal logical unit extracted from a :class:`LegalRule`.

    The ``guard`` field stores the canonical condition literals (e.g. ``token``
    or ``¬token``) while ``outcome`` captures the modality/action pair.  The
    Z3 fields are populated lazily by :func:`compile_atoms_to_z3`.
    """

    guard: Sequence[str]
    outcome: Mapping[str, str]
    source_id: str
    statute: str = ""
    section: str = ""
    text: str = ""
    modality: str = ""
    action: str = ""
    rule_type: str = "STATUTE"
    atom_id: str | None = None
    z3_guard: BoolRef | None = None
    z3_outcome: BoolRef | None = None

    @property
    def outcome_label(self) -> str:
        modality = self.outcome.get("modality", "UNKNOWN")
        action = self.outcome.get("action", "action")
        return f"{modality}:{action}"


def _literal_to_z3(literal: str, variables: MutableMapping[str, BoolRef]) -> BoolRef:
    """Return a Boolean variable for *literal* creating it if necessary."""

    negated = literal.startswith("¬")
    atom = literal[1:] if negated else literal
    if atom not in variables:
        variables[atom] = Bool(atom)
    return Not(variables[atom]) if negated else variables[atom]


def _outcome_to_z3(atom: PolicyAtom, variables: MutableMapping[str, BoolRef]) -> BoolRef:
    """Map an outcome into a Boolean assertion.

    ``OBLIGE`` is treated as a positive assertion on the action variable while
    ``FORBID`` is treated as a negated assertion.  ``PERMIT`` is modelled as a
    positive hint to keep the encoding consistent across modalities.
    """

    action_var = atom.outcome.get("action", atom.source_id)
    modality = atom.outcome.get("modality", "PERMIT").upper()
    if action_var not in variables:
        variables[action_var] = Bool(action_var)
    if modality == "FORBID":
        return Not(variables[action_var])
    return variables[action_var]


def _cache_key(manifest_hash: str | None, jurisdictions: Sequence[str] | None) -> Tuple[str, Tuple[str, ...]]:
    return (manifest_hash or "default", tuple(sorted(jurisdictions)) if jurisdictions else ("*all*",))


def build_policy_atoms_from_rules(
    rules: Iterable[LegalRule],
    mapper: PredicateMapper | None = None,
    manifest_hash: str | None = None,
    jurisdictions: Sequence[str] | None = None,
) -> List[PolicyAtom]:
    """Convert ``LegalRule`` objects into :class:`PolicyAtom` instances.

    Results are memoized by manifest hash and jurisdiction filter to avoid
    repetitive Z3 expression construction across hunts.
    """

    cache_key = _cache_key(manifest_hash, jurisdictions)
    if cache_key in _ATOM_CACHE:
        return list(_ATOM_CACHE[cache_key]["atoms"])

    atoms: List[PolicyAtom] = []
    jurisdictions_lower = {j.lower() for j in jurisdictions} if jurisdictions else None
    for idx, rule in enumerate(rules):
        if jurisdictions_lower and getattr(rule, "jurisdiction", "").lower() not in jurisdictions_lower:
            continue
        guard = list(rule.conditions)
        atom = PolicyAtom(
            guard=guard,
            outcome={
                "modality": rule.modality,
                "action": rule.action,
                "subject": rule.subject,
                "jurisdiction": getattr(rule, "jurisdiction", ""),
            },
            source_id=getattr(rule, "id", rule.action),
            statute=getattr(rule, "statute", ""),
            section=getattr(rule, "section", ""),
            text=getattr(rule, "text", ""),
            modality=rule.modality,
            action=rule.action,
            atom_id=f"{getattr(rule, 'id', rule.action)}_atom_{idx}",
        )
        atoms.append(atom)
    var_map: Dict[str, BoolRef] = {}
    compile_atoms_to_z3(atoms, var_map)
    base_optimize = Optimize()
    for atom in atoms:
        if atom.z3_guard is not None and atom.z3_outcome is not None:
            base_optimize.add(Implies(atom.z3_guard, atom.z3_outcome))
    _ATOM_CACHE[cache_key] = {"atoms": atoms, "var_map": var_map, "optimize": base_optimize}
    return list(atoms)


def compile_atoms_to_z3(
    atoms: Sequence[PolicyAtom],
    variables: MutableMapping[str, BoolRef] | None = None,
) -> Dict[str, BoolRef]:
    """Populate Z3 expressions for guards/outcomes and return the var map."""

    variables = variables or {}
    for atom in atoms:
        z3_guard_terms = [_literal_to_z3(lit, variables) for lit in atom.guard]
        atom.z3_guard = And(*z3_guard_terms) if z3_guard_terms else BoolVal(True)
        atom.z3_outcome = _outcome_to_z3(atom, variables)
    return variables


def memoized_optimize(
    manifest_hash: str | None,
    jurisdictions: Sequence[str] | None,
) -> tuple[Optimize, Dict[str, BoolRef]]:
    """Return a reusable Optimize seeded with cached rule implications."""

    cache_key = _cache_key(manifest_hash, jurisdictions)
    if cache_key not in _ATOM_CACHE:
        raise ValueError("Atoms must be built before requesting memoized optimize")
    cached = _ATOM_CACHE[cache_key]
    base_opt = cached.get("optimize")
    var_map = cached.get("var_map")
    if not isinstance(base_opt, Optimize) or not isinstance(var_map, dict):
        raise ValueError("Cache incomplete for optimize")
    clone = Optimize()
    for assertion in base_opt.assertions():
        clone.add(assertion)
    return clone, var_map  # type: ignore[return-value]


def build_z3_model_for_scenario(
    scenario: Mapping[str, bool], atoms: Sequence[PolicyAtom]
) -> tuple[Solver, Dict[str, BoolRef]]:
    """Instantiate a solver seeded with scenario facts and rule implications."""

    var_map: Dict[str, BoolRef] = {}
    compile_atoms_to_z3(atoms, var_map)
    solver = Solver()

    # Fix scenario atoms
    for name, value in scenario.items():
        if name not in var_map:
            var_map[name] = Bool(name)
        solver.add(var_map[name] == BoolVal(bool(value)))

    # Add guard ⇒ outcome implications
    for atom in atoms:
        if atom.z3_guard is None or atom.z3_outcome is None:
            continue
        solver.add(Implies(atom.z3_guard, atom.z3_outcome))

    return solver, var_map


def check_scenario(
    scenario: Mapping[str, bool], atoms: Sequence[PolicyAtom]
) -> tuple[bool, List[PolicyAtom]]:
    """Check if *scenario* is consistent with ``atoms``.

    Returns a tuple ``(is_sat, active_atoms)`` where ``active_atoms`` are those
    whose guards are satisfied under the provided facts.
    """

    with _Z3_LOCK:
        solver, var_map = build_z3_model_for_scenario(scenario, atoms)
        # Identify active atoms by evaluating guards under the model's assumptions
        active_atoms: List[PolicyAtom] = []
        for atom in atoms:
            if atom.z3_guard is None:
                continue
            # Create a temporary solver to evaluate guard truth under the scenario
            guard_solver = Solver()
            for assumption in solver.assertions():
                guard_solver.add(assumption)
            guard_solver.add(Not(atom.z3_guard))
            if guard_solver.check() == sat:
                continue
            active_atoms.append(atom)

        is_sat = solver.check() == sat
    return is_sat, active_atoms


def analyze_scenario(
    scenario: Mapping[str, bool], atoms: Sequence[PolicyAtom]
) -> tuple[bool, List[PolicyAtom], List[PolicyAtom]]:
    """Return satisfiability, active atoms, and the UNSAT core if any."""

    with _Z3_LOCK:
        var_map: Dict[str, BoolRef] = {}
        compile_atoms_to_z3(atoms, var_map)
        solver = Solver()
        solver.set(unsat_core=True)

        for name, value in scenario.items():
            if name not in var_map:
                var_map[name] = Bool(name)
            solver.add(var_map[name] == BoolVal(bool(value)))

        tracked_atoms: list[tuple[BoolRef, PolicyAtom]] = []
        for idx, atom in enumerate(atoms):
            if atom.z3_guard is None or atom.z3_outcome is None:
                continue
            tracker = Bool(f"atom_{idx}")
            solver.assert_and_track(Implies(atom.z3_guard, atom.z3_outcome), tracker)
            tracked_atoms.append((tracker, atom))

        sat_result = solver.check()
        unsat_atoms: list[PolicyAtom] = []
        if sat_result == unsat:
            core = set(solver.unsat_core())
            unsat_atoms = [atom for tracker, atom in tracked_atoms if tracker in core]

    is_sat, active_atoms = check_scenario(scenario, atoms)
    return is_sat, active_atoms, unsat_atoms
