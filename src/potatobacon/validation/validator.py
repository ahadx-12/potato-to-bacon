from __future__ import annotations
from typing import Dict, List
import sympy as sp
from sympy.core.expr import Expr
from potatobacon.models import Equation, ValidationReport
from .domains.registry import get_guard_for_domain
from .core_checks import run_basic_checks

def _build_symbols(assumptions: Dict[str, Dict[str, object]]) -> Dict[str, sp.Symbol]:
    out: Dict[str, sp.Symbol] = {}
    for name, opts in assumptions.items():
        opts = opts or {}
        # Filter only truthy SymPy assumptions; ignore unknown keys
        valid_kw = {}
        for k, v in opts.items():
            if v is True:
                valid_kw[k] = True
        out[name] = sp.Symbol(name, **valid_kw)
    return out

def validate_equation(equation: Equation) -> List[ValidationReport]:
    """
    End-to-end validation orchestration:
      1) Materialize assumption-aware symbols.
      2) Sympify RHS ONCE with locals mapping.
      3) Run general checks.
      4) Run domain guard checks.
    """
    reports: List[ValidationReport] = []

    # 1) Symbols
    try:
        symbols = _build_symbols(equation.symbol_assumptions)
    except TypeError as e:
        return [ValidationReport(
            check_name="symbol_instantiation_error",
            is_valid=False,
            message=f"Invalid assumptions: {e}",
            severity="ERROR"
        )]

    # 2) Parse RHS (assumption-aware)
    try:
        expr: Expr = sp.sympify(equation.rhs_str, locals=symbols)
    except Exception as e:
        return [ValidationReport(
            check_name="sympy_parse_error",
            is_valid=False,
            message=f"Failed to parse RHS '{equation.rhs_str}': {e}",
            severity="ERROR",
            extra={"rhs_str": equation.rhs_str}
        )]

    # 3) Basic checks (dimension completeness, numeric sampling, etc.)
    reports.extend(run_basic_checks(equation, symbols, expr))

    # 4) Domain-specific guard
    try:
        guard = get_guard_for_domain(equation.domain)
        reports.extend(guard.validate(equation, symbols, expr))
    except Exception as e:
        reports.append(ValidationReport(
            check_name="domain_guard_error",
            is_valid=False,
            message=f"Domain guard failure: {e}",
            severity="ERROR"
        ))

    return reports
