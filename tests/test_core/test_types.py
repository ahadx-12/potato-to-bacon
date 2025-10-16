"""Tests for core type definitions."""

import pytest
import sympy as sp

from potatobacon.core.dimensions import ENERGY, MASS, VELOCITY
from potatobacon.core.types import (
    Equation,
    EquationDomain,
    TranslationResult,
    ValidationResult,
    Variable,
    VariableRole,
)


def test_variable_creation():
    m = Variable("m", VariableRole.INPUT, MASS, unit="kg", description="mass")
    assert m.name == "m"
    assert m.role == VariableRole.INPUT
    assert m.unit == "kg"


def test_variable_constraints():
    m = Variable("m", VariableRole.INPUT, MASS, constraints={"positive": True})
    assert m.constraints["positive"] is True


def test_equation_creation():
    m = Variable("m", VariableRole.INPUT, MASS)
    v = Variable("v", VariableRole.INPUT, VELOCITY)
    ke = Variable("kinetic_energy", VariableRole.OUTPUT, ENERGY)
    eq = Equation(
        name="kinetic_energy",
        inputs=[m, v],
        outputs=[ke],
        expression=sp.sympify("0.5*m*v**2"),
        domain=EquationDomain.CLASSICAL,
        tags=["mechanics", "energy"],
    )
    assert eq.name == "kinetic_energy"
    assert len(eq.inputs) == 2
    assert eq.domain == EquationDomain.CLASSICAL
    assert "mechanics" in eq.tags


def test_equation_get_variable():
    m = Variable("m", VariableRole.INPUT, MASS)
    v = Variable("v", VariableRole.INPUT, VELOCITY)
    ke = Variable("ke", VariableRole.OUTPUT, ENERGY)
    eq = Equation("ke", [m, v], [ke], sp.sympify("0.5*m*v**2"))
    assert eq.get_variable("m") is m
    assert eq.get_variable("missing") is None


def test_validation_result():
    result = ValidationResult(valid=True)
    result.add_check_passed("dimensional_consistency")
    result.add_warning("Large velocity")
    assert result.valid
    assert result.checks_passed == ["dimensional_consistency"]
    assert result.warnings == ["Large velocity"]


def test_validation_result_with_error():
    result = ValidationResult(valid=True)
    result.add_error("Mismatch")
    assert not result.valid
    assert result.errors == ["Mismatch"]


def test_translation_result():
    m = Variable("m", VariableRole.INPUT, MASS)
    v = Variable("v", VariableRole.INPUT, VELOCITY)
    ke = Variable("kinetic_energy", VariableRole.OUTPUT, ENERGY)
    eq = Equation("kinetic_energy", [m, v], [ke], sp.sympify("0.5*m*v**2"))
    validation = ValidationResult(valid=True)
    validation.add_check_passed("dimensional_consistency")
    result = TranslationResult(
        equation=eq,
        canonical_form="kinetic_energy = 0.5*m*v**2",
        schema={"name": "kinetic_energy"},
        validation=validation,
        manifest_hash="abc123",
    )
    assert result.is_valid()
    assert "kinetic_energy" in result.summary()


def test_equation_requires_inputs():
    ke = Variable("ke", VariableRole.OUTPUT, ENERGY)
    with pytest.raises(ValueError):
        Equation("invalid", [], [ke], sp.sympify("42"))


def test_equation_requires_outputs():
    m = Variable("m", VariableRole.INPUT, MASS)
    with pytest.raises(ValueError):
        Equation("invalid", [m], [], sp.sympify("m"))
