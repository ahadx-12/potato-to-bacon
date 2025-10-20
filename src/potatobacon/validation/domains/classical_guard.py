from __future__ import annotations
from typing import Dict, List
import sympy as sp
from sympy.core.expr import Expr
from potatobacon.models import Equation, ValidationReport
from .base_guard import DomainGuard

class ClassicalGuard(DomainGuard):
    """
    Minimal baseline: add classical sanity checks as needed.
    Intentionally conservative—real work happens in general validator + domains.
    """
    def validate(self, equation: Equation, symbols: Dict[str, sp.Symbol], expr: Expr) -> List[ValidationReport]:
        reports: List[ValidationReport] = []
        # Example (placeholder): expression should be finite for generic symbol assumptions
        if expr.is_finite is False:
            reports.append(ValidationReport(
                check_name="classical_finiteness",
                is_valid=False,
                message="Expression is symbolically non-finite under declared assumptions.",
                severity="ERROR"
            ))
        return reports
