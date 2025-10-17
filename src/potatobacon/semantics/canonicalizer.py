"""Tools for converting parsed equations into a canonical form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import sympy as sp

from ..core.types import Equation, Variable
from .ir import TheoryIR


class Canonicalizer:
    """Produce a deterministic intermediate representation from an equation."""

    def canonicalize(self, equation: Equation) -> TheoryIR:
        if not isinstance(equation, Equation):
            raise TypeError("canonicalize expects an Equation instance")

        simplified = sp.simplify(equation.expression)
        canonical_expr = str(simplified)
        output_name = equation.outputs[0].name
        canonical_str = f"{output_name} = {canonical_expr}"

        variables: Dict[str, Variable] = {var.name: var for var in equation.all_variables()}

        return TheoryIR(
            equation=equation,
            simplified_expr=simplified,
            canonical_str=canonical_str,
            inputs=list(equation.inputs),
            outputs=list(equation.outputs),
            variables=variables,
        )


@dataclass
class CanonicalExpr:
    simplified_expr: sp.Basic
    canonical_str: str


def canonicalize(expr_or_eq: sp.Basic | sp.Equality) -> CanonicalExpr:
    """Canonicalize a raw SymPy expression or equality."""
    if isinstance(expr_or_eq, sp.Equality):
        simplified = sp.simplify(expr_or_eq.lhs - expr_or_eq.rhs)
        canonical_str = f"{str(expr_or_eq.lhs)} = {str(expr_or_eq.rhs)}"
    else:
        simplified = sp.simplify(expr_or_eq)
        canonical_str = str(simplified)
    return CanonicalExpr(simplified_expr=simplified, canonical_str=canonical_str)
