from __future__ import annotations
from typing import Dict, Any, List, Optional
import sympy as sp

from potatobacon.validation.dimensional import validate_dimensions  # existing
from potatobacon.validation.constraints import validate_constraints, ConstraintError
from potatobacon.validation.relativistic import validate_relativistic, RelativisticError
from potatobacon.validation.pde import classify_pde


def validate_all(
    expr_or_eq: sp.Basic | sp.Equality,
    units: Dict[str, str],
    result_unit: Optional[str],
    constraints: Dict[str, Any] | None = None,
    domain: str = "classical",
    pde_space_vars: Optional[List[str]] = None,
    pde_time_var: Optional[str] = None,
    checks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs the layered validator pipeline; returns a report dict.
    checks: subset of ["dimensional","constraints","relativistic","pde_class"]
    """
    checks = checks or ["dimensional", "constraints", "relativistic", "pde_class"]
    report: Dict[str, Any] = {"ok": True}
    errors: List[Dict[str, str]] = []
    details: Dict[str, Any] = {}
    report["errors"] = errors
    report["details"] = details

    # 1) Dimensional check (fast)
    if "dimensional" in checks:
        try:
            validate_dimensions(expr_or_eq, units, result_unit)
            details["dimensional"] = {"ok": True}
        except Exception as e:
            report["ok"] = False
            errors.append({"stage": "dimensional", "message": str(e)})
            return report  # fail fast

    # 2) Basic constraints (realness + symbol-level)
    if "constraints" in checks:
        try:
            expr = (
                expr_or_eq.lhs - expr_or_eq.rhs
                if isinstance(expr_or_eq, sp.Equality)
                else expr_or_eq
            )
            validate_constraints(expr, constraints or {}, require_real=True)
            details["constraints"] = {"ok": True}
        except ConstraintError as ce:
            report["ok"] = False
            errors.append({"stage": "constraints", "message": str(ce)})
            return report

    # 3) Relativistic guards (if requested)
    if domain == "relativistic" and "relativistic" in checks:
        try:
            expr = (
                expr_or_eq.lhs - expr_or_eq.rhs
                if isinstance(expr_or_eq, sp.Equality)
                else expr_or_eq
            )
            validate_relativistic(expr, units, constants={"c": "m/s"}, strict=True)
            details["relativistic"] = {"ok": True}
        except RelativisticError as re:
            report["ok"] = False
            errors.append({"stage": "relativistic", "message": str(re)})
            return report

    # 4) PDE class (optional)
    if "pde_class" in checks:
        space_syms = [sp.Symbol(s) for s in (pde_space_vars or [])]
        t_sym = sp.Symbol(pde_time_var) if pde_time_var else None
        pclass = classify_pde(expr_or_eq, space_syms, t_sym)
        details["pde_class"] = {"ok": True, "class": pclass.value}

    return report
