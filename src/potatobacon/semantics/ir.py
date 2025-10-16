"""Intermediate representation for canonical equations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import sympy as sp

from ..core.types import Equation, Variable


@dataclass
class TheoryIR:
    """Intermediate representation of an equation after canonicalisation."""

    equation: Equation
    simplified_expr: sp.Expr
    canonical_str: str
    inputs: List[Variable] = field(default_factory=list)
    outputs: List[Variable] = field(default_factory=list)
    variables: Dict[str, Variable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.simplified_expr, sp.Expr):
            self.simplified_expr = sp.sympify(self.simplified_expr)
        if not self.canonical_str:
            raise ValueError("Canonical string cannot be empty")
        if not self.inputs:
            self.inputs = list(self.equation.inputs)
        if not self.outputs:
            self.outputs = list(self.equation.outputs)
        if not self.variables:
            self.variables = {var.name: var for var in self.equation.all_variables()}

    @property
    def canonical_equation(self) -> str:
        return self.canonical_str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.equation.name,
            "domain": self.equation.domain.value,
            "canonical": self.canonical_str,
            "sympy": str(self.simplified_expr),
        }
