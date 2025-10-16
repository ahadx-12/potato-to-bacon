"""Tests for the canonicalizer."""

from __future__ import annotations

import pytest

from potatobacon.parser import DSLParser
from potatobacon.semantics import Canonicalizer
from potatobacon.semantics.ir import TheoryIR


def test_canonicalize_returns_theory_ir() -> None:
    parser = DSLParser()
    canonicalizer = Canonicalizer()
    dsl = """
@classical
def kinetic_energy(m: mass, v: velocity) -> energy:
    return 0.5 * m * v**2
"""
    equation = parser.parse(dsl)
    ir = canonicalizer.canonicalize(equation)

    assert isinstance(ir, TheoryIR)
    assert ir.canonical_str == "kinetic_energy = 0.5*m*v**2"
    assert str(ir.simplified_expr) == "0.5*m*v**2"


def test_canonicalizer_requires_equation_instance() -> None:
    canonicalizer = Canonicalizer()
    with pytest.raises(TypeError):
        canonicalizer.canonicalize("not an equation")  # type: ignore[arg-type]


def test_canonicalizer_is_deterministic() -> None:
    parser = DSLParser()
    canonicalizer = Canonicalizer()
    dsl = """
@classical
def momentum(m: mass, v: velocity) -> momentum:
    return m * v
"""
    ir1 = canonicalizer.canonicalize(parser.parse(dsl))
    ir2 = canonicalizer.canonicalize(parser.parse(dsl))

    assert ir1.canonical_str == ir2.canonical_str
    assert str(ir1.simplified_expr) == str(ir2.simplified_expr)
