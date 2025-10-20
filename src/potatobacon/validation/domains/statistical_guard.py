from __future__ import annotations
from typing import Dict, List
import sympy as sp
from sympy.core.expr import Expr
from potatobacon.models import Equation, ValidationReport
from .base_guard import DomainGuard

class StatisticalGuard(DomainGuard):
    """
    Heuristics:
      1) If 'T' appears, enforce T.is_positive (absolute temperature).
      2) If LHS tagged 'entropy', attempt S ≥ 0 symbolically; else INFO.
    """
    def validate(self, equation: Equation, symbols: Dict[str, sp.Symbol], expr: Expr) -> List[ValidationReport]:
        reports: List[ValidationReport] = []

        T = symbols.get("T")
        if T and T in expr.free_symbols and not T.is_positive:
            reports.append(ValidationReport(
                check_name="temperature_positive_required",
                is_valid=False,
                message="Temperature 'T' must have assumption positive=True for statistical mechanics.",
                severity="ERROR"
            ))

        if equation.symbol_metadata.get(equation.lhs_str) == "entropy":
            nonneg = sp.ask(sp.Q.nonnegative(expr))
            if nonneg is False:
                reports.append(ValidationReport(
                    check_name="entropy_negative",
                    is_valid=False,
                    message="Entropy expression proven negative; violates S ≥ 0.",
                    severity="ERROR"
                ))
            elif nonneg is None:
                reports.append(ValidationReport(
                    check_name="entropy_nonneg_inconclusive",
                    is_valid=True,
                    message="Could not prove S ≥ 0 symbolically. Use sampling step for confirmation.",
                    severity="INFO"
                ))

        return reports
