"""Tests for dimensional validation."""

from __future__ import annotations

import pytest

from potatobacon.parser import DSLParser
from potatobacon.semantics import Canonicalizer
from potatobacon.validation import DimensionalValidator


@pytest.fixture
def pipeline() -> dict:
    return {
        "parser": DSLParser(),
        "canonicalizer": Canonicalizer(),
        "validator": DimensionalValidator(),
    }


def test_valid_kinetic_energy(pipeline: dict) -> None:
    dsl = """
@classical
def kinetic_energy(m: mass, v: velocity) -> energy:
    return 0.5 * m * v**2
"""
    equation = pipeline["parser"].parse(dsl)
    ir = pipeline["canonicalizer"].canonicalize(equation)
    result = pipeline["validator"].validate(ir)

    assert result.valid
    assert "output_dimensions_match" in result.checks_passed


def test_invalid_dimension_mismatch(pipeline: dict) -> None:
    dsl = """
@classical
def invalid(m: mass, c: velocity) -> energy:
    return m * c**3
"""
    equation = pipeline["parser"].parse(dsl)
    ir = pipeline["canonicalizer"].canonicalize(equation)
    result = pipeline["validator"].validate(ir)

    assert not result.valid
    assert result.errors


def test_newton_second_law(pipeline: dict) -> None:
    dsl = """
@classical
def newton_second_law(m: mass, a: acceleration) -> force:
    return m * a
"""
    equation = pipeline["parser"].parse(dsl)
    ir = pipeline["canonicalizer"].canonicalize(equation)
    result = pipeline["validator"].validate(ir)

    assert result.valid
