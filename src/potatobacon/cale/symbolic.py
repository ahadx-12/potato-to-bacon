"""Symbolic conflict checker powered by Z3."""

from __future__ import annotations

from typing import List, Tuple

from z3 import And, Bool, BoolRef, BoolVal, Implies, Not, Solver, unsat, unknown

from .parser import PredicateMapper
from .types import LegalRule


class SymbolicConflictChecker:
    """Compute symbolic conflict intensity between two :class:`LegalRule` objects."""

    def __init__(self, predicate_mapper: PredicateMapper) -> None:
        self._predicate_mapper = predicate_mapper

    def _antecedent_bool(self, conditions: List[str]) -> BoolRef:
        if not conditions:
            return BoolVal(True)
        clauses: List[BoolRef] = []
        for literal in conditions:
            atom = literal.lstrip("¬")
            # Ensure the mapper records the atom for debugging parity.
            try:
                self._predicate_mapper.canonical_atom(atom)
            except Exception:  # pragma: no cover - defensive safeguard
                pass
            symbol = Bool(f"cond_{atom}")
            if literal.startswith("¬"):
                clauses.append(Not(symbol))
            else:
                clauses.append(symbol)
        return And(*clauses)

    def _vars_for_action(self, action: str) -> Tuple[BoolRef, BoolRef, BoolRef, BoolRef]:
        act = Bool(f"act_{action}")
        forb = Bool(f"forb_{action}")
        perm = Bool(f"perm_{action}")
        must = Bool(f"must_{action}")
        return act, forb, perm, must

    def _rule_to_formula(self, rule: LegalRule) -> BoolRef:
        antecedent = self._antecedent_bool(rule.conditions)
        act, forb, perm, must = self._vars_for_action(rule.action)
        if rule.modality == "OBLIGE":
            return Implies(antecedent, And(must, act))
        if rule.modality == "FORBID":
            return Implies(antecedent, And(forb, Not(act)))
        if rule.modality == "PERMIT":
            return Implies(antecedent, perm)
        raise ValueError(f"Unknown modality: {rule.modality}")

    def check_conflict(self, rule_a: LegalRule, rule_b: LegalRule) -> float:
        if rule_a.action != rule_b.action:
            return 0.0

        antecedent_a = self._antecedent_bool(rule_a.conditions)
        antecedent_b = self._antecedent_bool(rule_b.conditions)

        antecedent_solver = Solver()
        antecedent_solver.add(antecedent_a, antecedent_b)
        if antecedent_solver.check() == unsat:
            return 0.0

        solver = Solver()
        _, forb, perm, must = self._vars_for_action(rule_a.action)
        solver.add(perm == Not(forb))
        solver.add(Implies(must, Not(forb)))

        solver.add(self._rule_to_formula(rule_a))
        solver.add(self._rule_to_formula(rule_b))
        solver.add(antecedent_a)
        solver.add(antecedent_b)

        result = solver.check()
        if result == unknown:  # pragma: no cover - defensive programming
            raise RuntimeError("Solver returned unknown result")

        return 1.0 if result == unsat else 0.0


__all__ = ["SymbolicConflictChecker"]

