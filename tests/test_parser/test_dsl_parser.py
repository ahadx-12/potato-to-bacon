"""Tests for the DSL parser."""

import pytest
import sympy as sp

from potatobacon.core.dimensions import ENERGY, FORCE, MASS, VELOCITY
from potatobacon.core.types import EquationDomain
from potatobacon.parser import DSLParser, ParseError
from potatobacon.parser.dsl_parser import parse_dsl


@pytest.fixture
def parser() -> DSLParser:
    return DSLParser()


def test_parse_kinetic_energy(parser: DSLParser) -> None:
    dsl = """
@classical @mechanics
def kinetic_energy(m: mass, v: velocity) -> energy:
    return 0.5 * m * v**2
"""
    equation = parser.parse(dsl)
    assert equation.name == "kinetic_energy"
    assert equation.domain == EquationDomain.CLASSICAL
    assert "mechanics" in equation.tags
    assert len(equation.inputs) == 2
    assert equation.inputs[0].dimensions == MASS
    assert equation.inputs[1].dimensions == VELOCITY
    assert equation.outputs[0].dimensions == ENERGY


def test_parse_newton_second_law(parser: DSLParser) -> None:
    dsl = """
@classical @mechanics @dynamics
def newton_second_law(m: mass, a: acceleration) -> force:
    return m * a
"""
    equation = parser.parse(dsl)
    assert equation.name == "newton_second_law"
    assert equation.domain == EquationDomain.CLASSICAL
    assert "dynamics" in equation.tags
    assert equation.outputs[0].dimensions == FORCE


def test_parse_missing_decorator_defaults_to_classical(parser: DSLParser) -> None:
    dsl = """
def simple(x: length) -> length:
    return 2.0 * x
"""
    equation = parser.parse(dsl)
    assert equation.domain == EquationDomain.CLASSICAL


def test_parse_relativistic_domain(parser: DSLParser) -> None:
    dsl = """
@relativistic
def lorentz_factor(v: velocity) -> dimensionless:
    return 1.0 / (1.0 - (v / 3e8)**2)**0.5
"""
    equation = parser.parse(dsl)
    assert equation.domain == EquationDomain.RELATIVISTIC


def test_parse_expression(parser: DSLParser) -> None:
    dsl = """
@classical
def test_ops(a: energy, b: energy) -> energy:
    return (a + b) * 2.0 - a / b
"""
    equation = parser.parse(dsl)
    assert isinstance(equation.expression, sp.Expr)


def test_parse_unknown_dimension(parser: DSLParser) -> None:
    dsl = """
@classical
def invalid(x: unknown) -> energy:
    return x
"""
    with pytest.raises(ParseError):
        parser.parse(dsl)


def test_parse_invalid_syntax(parser: DSLParser) -> None:
    dsl = """
@classical
def invalid(m: mass -> energy:
    return m
"""
    with pytest.raises(ParseError):
        parser.parse(dsl)


def test_parse_power_operator(parser: DSLParser) -> None:
    dsl = """
@classical
def area(r: length) -> length:
    return 3.14159 * r**2
"""
    equation = parser.parse(dsl)
    assert equation.expression.has(sp.Pow)


def test_parse_negative_number(parser: DSLParser) -> None:
    dsl = """
@classical
def test_neg(x: energy) -> energy:
    return -0.5 * x
"""
    equation = parser.parse(dsl)
    assert equation.expression == sp.sympify("-0.5*x")


def test_accepts_assignment() -> None:
    e = parse_dsl("E = m*c^2")
    assert isinstance(e, sp.Equality)
    assert str(e.lhs) == "E"


def test_accepts_equality() -> None:
    e = parse_dsl("E == m*c**2")
    assert isinstance(e, sp.Equality)
    assert str(e.rhs) in {"c**2*m", "m*c**2"}


def test_accepts_return_residual() -> None:
    e = parse_dsl("return E - m*c**2")
    assert isinstance(e, sp.Basic)
    assert e.has(sp.Symbol("E"))


def test_accepts_bare_expr() -> None:
    e = parse_dsl("0.5*m*v**2")
    assert isinstance(e, sp.Basic)
    assert {s.name for s in e.free_symbols} == {"m", "v"}
