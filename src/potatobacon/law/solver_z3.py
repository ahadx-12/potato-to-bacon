"""Z3 helpers for translating CALE policy atoms into constraints.

This module purposely reuses the existing CALE rule representation so that the
legal analytics endpoints can reason about ambiguity and contradictions without
introducing a new DSL.  The helpers here are intentionally lightweight: they
compile the familiar ``LegalRule`` objects into :class:`z3.BoolRef` guards and
outcomes that can be composed by downstream search/metrics components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from z3 import (  # type: ignore[import-not-found]
    And,
    Bool,
    BoolRef,
    BoolVal,
    Implies,
    Not,
    Solver,
    sat,
)

from potatobacon.cale.parser import PredicateMapper
from potatobacon.cale.types import LegalRule


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


def build_policy_atoms_from_rules(
    rules: Iterable[LegalRule], mapper: PredicateMapper | None = None
) -> List[PolicyAtom]:
    """Convert ``LegalRule`` objects into :class:`PolicyAtom` instances."""

    atoms: List[PolicyAtom] = []
    for idx, rule in enumerate(rules):
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
    return atoms


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
