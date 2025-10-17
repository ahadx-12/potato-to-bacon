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
        from sympy import Number, Symbol

        if isinstance(expr, Number):
            return Dimension()

        if isinstance(expr, Symbol):
            if expr in var_dims:
                return var_dims[expr]
            raise DimensionalError(f"Unknown symbol encountered: {expr}")

        if expr.is_Add:
            expected: Dimension | None = None
            for term in expr.args:
                term_dim = self._compute_dimensions(term, var_dims)
                if expected is None:
                    expected = term_dim
                elif expected != term_dim:
                    raise DimensionalError(f"Cannot add dimensions {expected} and {term_dim}")
            return expected or Dimension()

        if expr.is_Mul:
            result = Dimension()
            for factor in expr.args:
                result = result * self._compute_dimensions(factor, var_dims)
            return result

        if expr.is_Pow:
            base_dim = self._compute_dimensions(expr.base, var_dims)
            exponent = expr.exp
            if not exponent.is_number:
                raise DimensionalError(f"Exponent must be numeric, got {exponent}")
            if exponent.is_integer is False:
                raise DimensionalError("Exponent must be an integer for dimensional analysis")
            return base_dim ** int(exponent)

        raise DimensionalError(f"Unsupported expression type: {type(expr)!r}")


def validate_dimensions(expr_or_eq: sp.Basic | sp.Equality,
                        units: Dict[str, str],
                        result_unit: str | None) -> None:
    """Lightweight helper for pipeline use: ensure all symbols have declared units."""
    expr = expr_or_eq.lhs - expr_or_eq.rhs if isinstance(expr_or_eq, sp.Equality) else expr_or_eq
    missing = [str(sym) for sym in sorted(expr.free_symbols, key=lambda s: s.name) if str(sym) not in units]
    if missing:
        raise ValueError(f"Missing units for symbols: {', '.join(missing)}")

    if result_unit and result_unit not in units.values():
        # Allow result unit to be specified separately even if not tied to a symbol.
        pass
