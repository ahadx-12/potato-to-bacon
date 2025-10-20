from __future__ import annotations
from typing import Dict, List
import sympy as sp
from sympy.core.expr import Expr
from potatobacon.models import Equation, ValidationReport
from .base_guard import DomainGuard

# We avoid importing sympy.physics.units (slow); rely on metadata & integers.
class QuantumGuard(DomainGuard):
    """
    Heuristics:
      1) If LHS is tagged 'probability', try to prove nonnegative; else INFO.
      2) If LHS tagged 'energy', require dependence on at least one integer symbol (quantization).
      3) If any input symbol is tagged 'probability', ensure assumptions enforce real & nonnegative.
    """
    def validate(self, equation: Equation, symbols: Dict[str, sp.Symbol], expr: Expr) -> List[ValidationReport]:
        reports: List[ValidationReport] = []

        # (1) Output probability non-negativity (symbolic attempt)
        if equation.symbol_metadata.get(equation.lhs_str) == "probability":
            nonneg = sp.ask(sp.Q.nonnegative(expr))
            if nonneg is False:
                reports.append(ValidationReport(
                    check_name="quantum_output_probability_negative",
                    is_valid=False,
                    message=f"Output '{equation.lhs_str}' is proven negative; violates probability constraints.",
                    severity="ERROR"
                ))
            elif nonneg is None:
                reports.append(ValidationReport(
                    check_name="quantum_output_probability_inconclusive",
                    is_valid=True,
                    message=f"Cannot symbolically prove '{equation.lhs_str} ≥ 0'. Consider numeric sampling.",
                    severity="INFO"
                ))

        # (2) Energy quantization heuristic
        if equation.symbol_metadata.get(equation.lhs_str) == "energy":
            integer_syms = [s for s in expr.free_symbols if s.is_integer is True]
            if not integer_syms:
                reports.append(ValidationReport(
                    check_name="energy_quantization_missing_integer_symbol",
                    is_valid=False,
                    message="Energy does not depend on any integer quantum number (e.g., n, l, m).",
                    severity="WARNING"
                ))

        # (3) Input probabilities must be real & nonnegative by assumptions
        for name, tag in equation.symbol_metadata.items():
            if tag == "probability" and name in symbols and name != equation.lhs_str:
                sym = symbols[name]
                if not (sym.is_real is True and (sym.is_nonnegative is True or sym.is_nonnegative is None and sym.is_positive is True)):
                    reports.append(ValidationReport(
                        check_name=f"prob_input_assumptions::{name}",
                        is_valid=False,
                        message=f"Symbol '{name}' tagged as probability must be real and nonnegative in assumptions.",
                        severity="ERROR"
                    ))

        return reports
