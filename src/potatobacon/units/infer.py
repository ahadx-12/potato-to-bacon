"""Unit inference utilities based on symbolic equations."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Tuple
import re

import sympy as sp

from potatobacon.units.algebra import (
    BASE_ORDER,
    DEFAULT_REGISTRY,
    Quantity,
    format_quantity,
    parse_unit_expr,
)


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED_NAMES = {
    "sin",
    "cos",
    "tan",
    "exp",
    "sqrt",
    "log",
    "Eq",
    "d",
    "d2",
}


@dataclass(slots=True)
class InferenceStep:
    """Structured reasoning step for inference traces."""

    action: str
    detail: str
    data: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload = {"action": self.action, "detail": self.detail}
        payload.update(self.data)
        return payload


class UnitInferenceError(RuntimeError):
    """Raised when inference fails due to inconsistent constraints."""


def _build_dimension_system(
    expr_src: str,
    known_units: Dict[str, str] | None,
    registry,
) -> Tuple[
    sp.Basic,
    Dict[str, Quantity],
    Dict[str, str],
    Dict[str, Tuple[sp.Symbol, ...]],
    List[sp.Expr],
    List[Tuple[Tuple[sp.Expr, ...], Tuple[sp.Expr, ...], sp.Basic]],
    List[InferenceStep],
    Tuple[sp.Expr, ...],
]:
    trace: List[InferenceStep] = []
    known_units = known_units or {}
    locals_map: Dict[str, sp.Symbol] = {}
    for name in known_units:
        locals_map[name] = sp.Symbol(name)

    source = expr_src
    if isinstance(expr_src, str):
        expr_str = expr_src.strip()
        if "==" not in expr_str and "=" in expr_str:
            left, right = expr_str.split("=", 1)
            expr_str = f"Eq({left.strip()}, {right.strip()})"
        for token in _TOKEN_RE.findall(expr_str):
            if token in _RESERVED_NAMES:
                continue
            locals_map.setdefault(token, sp.Symbol(token))
        source = expr_str

    expr = sp.sympify(source, locals=locals_map, evaluate=False)

    known_quantities: Dict[str, Quantity] = {}
    for symbol, unit_text in known_units.items():
        unit_text = unit_text.strip()
        if not unit_text or unit_text == "?":
            continue
        quantity = parse_unit_expr(unit_text, registry=registry)
        known_quantities[symbol] = quantity
        trace.append(
            InferenceStep(
                action="known",
                detail=f"Provided unit for {symbol}",
                data={"unit": format_quantity(quantity, registry=registry)},
            )
        )

    equations: List[sp.Expr] = []
    unknown_components: Dict[str, Tuple[sp.Symbol, ...]] = {}
    equality_pairs: List[Tuple[Tuple[sp.Expr, ...], Tuple[sp.Expr, ...], sp.Basic]] = []
    zero = tuple(sp.Integer(0) for _ in BASE_ORDER)

    def quantity_to_sympy(q: Quantity) -> Tuple[sp.Rational, ...]:
        return tuple(sp.Rational(d.numerator, d.denominator) for d in q.dims)

    def ensure_unknown(symbol: str) -> Tuple[sp.Symbol, ...]:
        if symbol not in unknown_components:
            unknown_components[symbol] = tuple(sp.Symbol(f"{symbol}_{axis}") for axis in BASE_ORDER)
        return unknown_components[symbol]

    def add_constraint(reason: str, lhs: Tuple[sp.Expr, ...], rhs: Tuple[sp.Expr, ...], *, context: Dict[str, object] | None = None) -> None:
        context = context or {}
        axis_data = {}
        for axis_name, left, right in zip(BASE_ORDER, lhs, rhs):
            expr_cmp = sp.simplify(left - right)
            if expr_cmp == 0:
                continue
            equations.append(expr_cmp)
            axis_data[axis_name] = sp.sstr(expr_cmp)
        if axis_data:
            payload = {"axes": axis_data}
            payload.update(context)
            trace.append(InferenceStep(action="constraint", detail=reason, data=payload))

    def eval_dims(node: sp.Basic) -> Tuple[sp.Expr, ...]:
        if isinstance(node, sp.Equality):
            lhs_dims = eval_dims(node.lhs)
            rhs_dims = eval_dims(node.rhs)
            equality_pairs.append((lhs_dims, rhs_dims, node))
            add_constraint(
                "Equality enforces identical dimensions",
                lhs_dims,
                rhs_dims,
                context={"equation": sp.sstr(node)},
            )
            return lhs_dims

        if node.is_Number:
            return zero

        if node.is_Symbol:
            name = str(node)
            if name in known_quantities:
                return quantity_to_sympy(known_quantities[name])
            return ensure_unknown(name)

        if node.is_Add:
            term_dims = [eval_dims(arg) for arg in node.args]
            first = term_dims[0]
            for other, term in zip(term_dims[1:], node.args[1:]):
                add_constraint(
                    "Sum terms must share dimensions",
                    first,
                    other,
                    context={"term": sp.sstr(term)},
                )
            return first

        if node.is_Mul:
            total = list(zero)
            for arg in node.args:
                dims = eval_dims(arg)
                total = [sp.simplify(a + b) for a, b in zip(total, dims)]
            return tuple(total)

        if node.is_Pow:
            base_dims = eval_dims(node.base)
            exponent = node.exp
            if exponent.is_number:
                exp_value = sp.nsimplify(exponent)
                return tuple(sp.simplify(d * exp_value) for d in base_dims)
            raise UnitInferenceError(f"Unsupported exponent '{sp.sstr(exponent)}' in power")

        if isinstance(node, sp.Derivative):
            base = list(eval_dims(node.expr))
            for sym, count in node.variable_count:
                var_dims = eval_dims(sym)
                base = [sp.simplify(b - count * v) for b, v in zip(base, var_dims)]
            return tuple(base)

        if node.is_Function:
            for arg in node.args:
                arg_dims = eval_dims(arg)
                add_constraint(
                    f"Function '{node.func.__name__}' expects dimensionless arguments",
                    arg_dims,
                    zero,
                    context={"argument": sp.sstr(arg)},
                )
            return zero

        raise UnitInferenceError(f"Unsupported expression element: {type(node).__name__}")

    expr_dims = eval_dims(expr)

    canonical_known = {
        name: format_quantity(quantity, registry=registry)
        for name, quantity in sorted(known_quantities.items())
    }

    return expr, known_quantities, canonical_known, unknown_components, equations, equality_pairs, trace, expr_dims


def infer_from_equation(
    expr_src: str,
    known_units: Dict[str, str] | None = None,
    *,
    registry=DEFAULT_REGISTRY,
) -> Tuple[Dict[str, str], List[Dict[str, object]]]:
    """Infer missing symbol units from ``expr_src``."""

    (
        expr,
        known_quantities,
        canonical_known,
        unknown_components,
        equations,
        equality_pairs,
        trace,
        _,
    ) = _build_dimension_system(expr_src, known_units, registry)

    unknown_symbols = sorted(unknown_components.keys())
    output_map = dict(canonical_known)

    if not unknown_symbols:
        trace.append(InferenceStep(action="noop", detail="No unknown units to infer.", data={}))
        return output_map, [step.to_dict() for step in trace]

    linear_vars: List[sp.Symbol] = []
    offsets: Dict[str, int] = {}
    for name in unknown_symbols:
        offsets[name] = len(linear_vars)
        linear_vars.extend(unknown_components[name])

    if not equations:
        trace.append(
            InferenceStep(
                action="underdetermined",
                detail="Equation does not impose dimensional constraints on unknown symbols.",
                data={"symbols": unknown_symbols},
            )
        )
        return output_map, [step.to_dict() for step in trace]

    solution_set = sp.linsolve(equations, linear_vars)
    if not solution_set:
        trace.append(
            InferenceStep(
                action="fail",
                detail="No unit assignment satisfies all dimensional constraints.",
                data={},
            )
        )
        return output_map, [step.to_dict() for step in trace]

    solution = next(iter(solution_set))
    if any(expr.free_symbols for expr in solution):
        trace.append(
            InferenceStep(
                action="underdetermined",
                detail="Insufficient constraints to determine unique units.",
                data={"symbols": unknown_symbols},
            )
        )
        return output_map, [step.to_dict() for step in trace]

    sol_map = {var: value for var, value in zip(linear_vars, solution)}

    def to_fraction(expr: sp.Expr) -> Fraction:
        simplified = sp.nsimplify(expr)
        if simplified.is_Rational:
            return Fraction(int(simplified.p), int(simplified.q))
        if simplified.is_Number:
            return Fraction(float(simplified)).limit_denominator(10_000)
        raise UnitInferenceError(f"Non-numeric dimension encountered: {sp.sstr(expr)}")

    for name in unknown_symbols:
        start = offsets[name]
        comp = solution[start : start + len(BASE_ORDER)]
        dims = tuple(to_fraction(component) for component in comp)
        inferred_quantity = Quantity(1.0, dims)
        output_map[name] = format_quantity(inferred_quantity, registry=registry)
        trace.append(
            InferenceStep(
                action="infer",
                detail=f"Solved dimensions for {name}",
                data={
                    "dims": {axis: (int(d) if d.denominator == 1 else str(d)) for axis, d in zip(BASE_ORDER, dims)},
                    "unit": output_map[name],
                },
            )
        )

    # Provide overall dimension summary for Eq expressions
    if equality_pairs:
        for lhs_dims_expr, rhs_dims_expr, eq_node in equality_pairs:
            def substitute_dims(dims_expr: Tuple[sp.Expr, ...]) -> Tuple[Fraction, ...]:
                numeric = []
                for component in dims_expr:
                    component = component.subs(sol_map)
                    numeric.append(to_fraction(component))
                return tuple(numeric)

            lhs_dims = substitute_dims(lhs_dims_expr)
            rhs_dims = substitute_dims(rhs_dims_expr)
            lhs_quantity = Quantity(1.0, lhs_dims)
            rhs_quantity = Quantity(1.0, rhs_dims)
            trace.append(
                InferenceStep(
                    action="summary",
                    detail=f"Equality '{sp.sstr(eq_node)}'",
                    data={
                        "lhs": format_quantity(lhs_quantity, registry=registry),
                        "rhs": format_quantity(rhs_quantity, registry=registry),
                    },
                )
            )

    return output_map, [step.to_dict() for step in trace]


def evaluate_equation_dimensions(
    expr_src: str,
    known_units: Dict[str, str] | None = None,
    *,
    registry=DEFAULT_REGISTRY,
) -> Tuple[Dict[str, str], List[Dict[str, object]], List[Dict[str, str]]]:
    """Check dimensional consistency of ``expr_src`` using provided units."""

    (
        expr,
        _known_quantities,
        canonical_known,
        unknown_components,
        equations,
        equality_pairs,
        trace,
        expr_dims,
    ) = _build_dimension_system(expr_src, known_units, registry)

    unknown_symbols = sorted(unknown_components.keys())
    if unknown_symbols:
        raise UnitInferenceError(
            "Missing units for symbols: " + ", ".join(unknown_symbols)
        )

    def to_fraction(expr_val: sp.Expr) -> Fraction:
        simplified = sp.nsimplify(expr_val)
        if simplified.is_Rational:
            return Fraction(int(simplified.p), int(simplified.q))
        if simplified.is_Number:
            return Fraction(float(simplified)).limit_denominator(10_000)
        raise UnitInferenceError(f"Unable to resolve dimension component: {sp.sstr(expr_val)}")

    inconsistent = [sp.simplify(eq) for eq in equations if sp.simplify(eq) != 0]
    if inconsistent:
        raise UnitInferenceError(
            "Dimension mismatch detected: "
            + ", ".join(sp.sstr(expr) for expr in inconsistent)
        )

    summaries: List[Dict[str, str]] = []
    for lhs_dims_expr, rhs_dims_expr, eq_node in equality_pairs:
        lhs_dims = tuple(to_fraction(component) for component in lhs_dims_expr)
        rhs_dims = tuple(to_fraction(component) for component in rhs_dims_expr)
        lhs_quantity = Quantity(1.0, lhs_dims)
        rhs_quantity = Quantity(1.0, rhs_dims)
        summaries.append(
            {
                "equation": sp.sstr(eq_node),
                "lhs": format_quantity(lhs_quantity, registry=registry),
                "rhs": format_quantity(rhs_quantity, registry=registry),
            }
        )

    # For expressions that are not explicit equalities, report the total dimension.
    if not equality_pairs:
        expr_quantity = Quantity(1.0, tuple(to_fraction(component) for component in expr_dims))
        summaries.append(
            {
                "equation": sp.sstr(expr),
                "lhs": format_quantity(expr_quantity, registry=registry),
                "rhs": format_quantity(expr_quantity, registry=registry),
            }
        )

    return canonical_known, [step.to_dict() for step in trace], summaries


__all__ = [
    "infer_from_equation",
    "evaluate_equation_dimensions",
    "InferenceStep",
    "UnitInferenceError",
]

