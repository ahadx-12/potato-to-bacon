"""Dimensional validation utilities."""

from __future__ import annotations

from typing import Dict

import sympy as sp

from ..core.dimensions import Dimension, DimensionalError
from ..core.types import ValidationResult
from ..semantics.ir import TheoryIR


class DimensionalValidator:
    """Validate dimensional consistency of an equation's intermediate representation."""

    def validate(self, ir: TheoryIR) -> ValidationResult:
        """Run dimensional checks on the provided theory IR."""
        result = ValidationResult(valid=True)

        try:
            self._check_output_dimensions(ir)
            result.add_check_passed("output_dimensions_match")
        except DimensionalError as exc:
            result.add_error(f"Dimensional error: {exc}")

        return result

    def _check_output_dimensions(self, ir: TheoryIR) -> None:
        """Ensure the expression's dimensions match the declared output dimensions."""
        if not ir.outputs:
            raise DimensionalError("Theory IR must contain at least one output variable")

        var_dims: Dict[sp.Symbol, Dimension] = {}
        for variable in ir.variables.values():
            var_dims[sp.Symbol(variable.name)] = variable.dimensions

        computed = self._compute_dimensions(ir.simplified_expr, var_dims)
        declared = ir.outputs[0].dimensions

        if computed != declared:
            raise DimensionalError(
                f"Output dimension mismatch: computed {computed} but declared {declared}"
            )

    def _compute_dimensions(self, expr: sp.Expr, var_dims: Dict[sp.Symbol, Dimension]) -> Dimension:
        """Recursively determine the dimensions of a symbolic expression."""
        from sympy import Add, Div, Mul, Number, Pow, Symbol

        if isinstance(expr, Number):
            return Dimension()

        if isinstance(expr, Symbol):
            if expr in var_dims:
                return var_dims[expr]
            raise DimensionalError(f"Unknown symbol encountered: {expr}")

        if isinstance(expr, Add):
            expected: Dimension | None = None
            for term in expr.args:
                term_dim = self._compute_dimensions(term, var_dims)
                if expected is None:
                    expected = term_dim
                elif expected != term_dim:
                    raise DimensionalError(f"Cannot add dimensions {expected} and {term_dim}")
            return expected or Dimension()

        if isinstance(expr, Mul):
            result = Dimension()
            for factor in expr.args:
                result = result * self._compute_dimensions(factor, var_dims)
            return result

        if isinstance(expr, Div):
            numerator = self._compute_dimensions(expr.args[0], var_dims)
            denominator = self._compute_dimensions(expr.args[1], var_dims)
            return numerator / denominator

        if isinstance(expr, Pow):
            base_dim = self._compute_dimensions(expr.args[0], var_dims)
            exponent = expr.args[1]
            if not isinstance(exponent, Number):
                raise DimensionalError(f"Exponent must be numeric, got {exponent}")
            exp_value = exponent.value
            if abs(exp_value - round(exp_value)) > 1e-9:
                raise DimensionalError("Exponent must be an integer for dimensional analysis")
            return base_dim ** int(round(exp_value))

        raise DimensionalError(f"Unsupported expression type: {type(expr)!r}")
