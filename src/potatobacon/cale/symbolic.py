"""Symbolic conflict checker powered by Z3."""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - maintain deterministic fallback
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from z3 import And, Bool, BoolRef, BoolVal, Implies, Not, Solver, unsat, unknown

    _Z3_AVAILABLE = True
except Exception:  # pragma: no cover - lightweight fallback
    And = Bool = BoolRef = BoolVal = Implies = Not = Solver = None  # type: ignore[assignment]
    unsat = unknown = object()  # type: ignore[assignment]
    _Z3_AVAILABLE = False

SEED = int(os.getenv("CALE_SEED", "1337"))
random.seed(SEED)
np.random.seed(SEED)
if torch is not None:  # pragma: no branch
    try:  # pragma: no cover - guard against broken torch installs
        torch.manual_seed(SEED)
    except Exception:
        pass

from .parser import PredicateMapper
from .types import LegalRule


class SymbolicConflictChecker:
    """Compute symbolic conflict intensity between two :class:`LegalRule` objects."""

    def __init__(self, predicate_mapper: PredicateMapper) -> None:
        self._predicate_mapper = predicate_mapper

    def _antecedent_bool(self, conditions: List[str]) -> BoolRef:
        if not _Z3_AVAILABLE:
            raise RuntimeError("Z3 backend unavailable")
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
        if not _Z3_AVAILABLE:
            raise RuntimeError("Z3 backend unavailable")
        act = Bool(f"act_{action}")
        forb = Bool(f"forb_{action}")
        perm = Bool(f"perm_{action}")
        must = Bool(f"must_{action}")
        return act, forb, perm, must

    def _rule_to_formula(self, rule: LegalRule) -> BoolRef:
        if not _Z3_AVAILABLE:
            raise RuntimeError("Z3 backend unavailable")
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
        if not _Z3_AVAILABLE:
            return float(self._check_conflict_fallback(rule_a, rule_b))
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

    def _check_conflict_fallback(self, rule_a: LegalRule, rule_b: LegalRule) -> bool:
        if rule_a.action != rule_b.action:
            return False

        def _split(conditions: List[str]) -> Tuple[set[str], set[str]]:
            positives: set[str] = set()
            negatives: set[str] = set()
            for literal in conditions:
                atom = literal.lstrip("¬")
                (negatives if literal.startswith("¬") else positives).add(atom)
            return positives, negatives

        pos_a, neg_a = _split(rule_a.conditions)
        pos_b, neg_b = _split(rule_b.conditions)

        if (pos_a & neg_b) or (pos_b & neg_a):
            return False

        conflict_pairs = {
            ("OBLIGE", "FORBID"),
            ("FORBID", "OBLIGE"),
            ("FORBID", "PERMIT"),
            ("PERMIT", "FORBID"),
        }
        return (rule_a.modality, rule_b.modality) in conflict_pairs


__all__ = ["SymbolicConflictChecker"]

