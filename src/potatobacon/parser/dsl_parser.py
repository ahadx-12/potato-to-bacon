"""Parser for the potato.to.bacon domain specific language."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import sympy as sp

from ..core.dimensions import (
    ACCELERATION,
    DIMENSIONLESS,
    ENERGY,
    FORCE,
    LENGTH,
    MASS,
    MOMENTUM,
    POWER,
    TIME,
    VELOCITY,
    Dimension,
)
from ..core.types import Equation, EquationDomain, Variable, VariableRole


class ParseError(Exception):
    """Raised when DSL parsing fails."""


DIMENSION_MAP: Dict[str, Dimension] = {
    "mass": MASS,
    "length": LENGTH,
    "time": TIME,
    "velocity": VELOCITY,
    "acceleration": ACCELERATION,
    "force": FORCE,
    "energy": ENERGY,
    "power": POWER,
    "momentum": MOMENTUM,
    "dimensionless": DIMENSIONLESS,
}


FUNC_DEF_PATTERN = re.compile(
    r"def\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<params>[^)]*)\)\s*->\s*(?P<rtype>[a-zA-Z_][a-zA-Z0-9_]*)\s*:"
)


class DSLParser:
    """Concrete parser that processes the small physics DSL without external deps."""

    def __init__(self) -> None:
        pass

    def parse(self, dsl_text: str) -> Equation:
        lines = [line.strip() for line in dsl_text.splitlines() if line.strip()]
        if not lines:
            raise ParseError("Empty DSL input")

        decorators: List[str] = []
        body_lines: List[str] = []
        func_line: Optional[str] = None
        for line in lines:
            if line.startswith("@") and func_line is None:
                for token in line.split():
                    if token.startswith("@"):
                        decorators.append(token[1:])
                continue
            if func_line is None:
                func_line = line
            else:
                body_lines.append(line)

        if func_line is None:
            raise ParseError("No function definition found")

        match = FUNC_DEF_PATTERN.match(func_line)
        if not match:
            raise ParseError(f"Invalid function definition: {func_line}")

        name = match.group("name")
        params_raw = match.group("params").strip()
        return_type_name = match.group("rtype")

        parameters: List[Variable] = []
        if params_raw:
            for part in params_raw.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ParseError(f"Invalid parameter definition: {part}")
                param_name, param_type = [segment.strip() for segment in part.split(":", 1)]
                dimension = DIMENSION_MAP.get(param_type)
                if dimension is None:
                    raise ParseError(f"Unknown dimension type: {param_type}")
                parameters.append(
                    Variable(name=param_name, role=VariableRole.INPUT, dimensions=dimension)
                )

        return_dimension = DIMENSION_MAP.get(return_type_name)
        if return_dimension is None:
            raise ParseError(f"Unknown dimension type: {return_type_name}")

        expression_line = None
        for line in body_lines:
            if line.startswith("return "):
                expression_line = line[len("return ") :]
                break
        if expression_line is None:
            raise ParseError("Function body must contain a return statement")

        try:
            expression = sp.simplify(sp.sympify(expression_line))
        except Exception as exc:  # pragma: no cover - defensive
            raise ParseError(f"Failed to parse expression: {exc}") from exc

        domain = EquationDomain.CLASSICAL
        tags: List[str] = []
        for deco in decorators:
            lowered = deco.lower()
            if lowered in EquationDomain._value2member_map_:
                domain = EquationDomain(lowered)
            else:
                tags.append(lowered)

        output_variable = Variable(name=name, role=VariableRole.OUTPUT, dimensions=return_dimension)

        return Equation(
            name=name,
            inputs=parameters,
            outputs=[output_variable],
            expression=expression,
            domain=domain,
            tags=tags,
            source="dsl",
        )


def parse_dsl(dsl_text: str) -> sp.Basic:
    """Parse DSL text into a SymPy expression or equality."""
    lines = [line.strip() for line in dsl_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty DSL text")

    # Look for return statement
    expr_line = None
    for line in reversed(lines):
        if line.startswith("return "):
            expr_line = line[len("return ") :]
            break
    if expr_line is None:
        raise ValueError("DSL must contain a return statement")

    local_ns = {"d2": lambda expr, var: sp.Function("d2")(expr, var)}

    if "==" in expr_line:
        lhs, rhs = expr_line.split("==", 1)
        return sp.Eq(sp.sympify(lhs, locals=local_ns), sp.sympify(rhs, locals=local_ns))

    return sp.sympify(expr_line, locals=local_ns)
