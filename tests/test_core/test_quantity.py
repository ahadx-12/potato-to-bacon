"""Tests for PhysicalQuantity operations."""

import sympy as sp
import pytest

from potatobacon.core.dimensions import ENERGY, LENGTH, MASS, VELOCITY, DimensionalError
from potatobacon.core.quantity import PhysicalQuantity


def test_quantity_creation():
    mass = PhysicalQuantity("m", sp.Symbol("m"), MASS, unit="kg")
    assert mass.symbol == "m"
    assert mass.unit == "kg"
    assert mass.dimensions == MASS


def test_quantity_multiplication():
    m = PhysicalQuantity("m", sp.Symbol("m"), MASS)
    v = PhysicalQuantity("v", sp.Symbol("v"), VELOCITY)
    momentum = m * v
    assert momentum.dimensions.mass == 1
    assert momentum.dimensions.length == 1
    assert momentum.dimensions.time == -1


def test_quantity_division():
    energy = PhysicalQuantity("E", sp.Symbol("E"), ENERGY)
    mass = PhysicalQuantity("m", sp.Symbol("m"), MASS)
    result = energy / mass
    assert result.dimensions.mass == 0
    assert result.dimensions.length == 2
    assert result.dimensions.time == -2


def test_quantity_power():
    velocity = PhysicalQuantity("v", sp.Symbol("v"), VELOCITY)
    squared = velocity**2
    assert squared.dimensions.length == 2
    assert squared.dimensions.time == -2


def test_quantity_addition_same_dimensions():
    x1 = PhysicalQuantity("x1", sp.Symbol("x1"), LENGTH)
    x2 = PhysicalQuantity("x2", sp.Symbol("x2"), LENGTH)
    total = x1 + x2
    assert total.dimensions == LENGTH
    assert sp.simplify(total.expression - (x1.expression + x2.expression)) == 0


def test_quantity_addition_different_dimensions_fails():
    length = PhysicalQuantity("L", sp.Symbol("L"), LENGTH)
    mass = PhysicalQuantity("m", sp.Symbol("m"), MASS)
    with pytest.raises(DimensionalError):
        _ = length + mass


def test_quantity_subtraction_same_dimensions():
    x1 = PhysicalQuantity("x1", sp.Symbol("x1"), LENGTH)
    x2 = PhysicalQuantity("x2", sp.Symbol("x2"), LENGTH)
    diff = x1 - x2
    assert diff.dimensions == LENGTH


def test_dimensionless_check():
    ratio = PhysicalQuantity("r", sp.Symbol("r"), LENGTH) / PhysicalQuantity(
        "s", sp.Symbol("s"), LENGTH
    )
    assert ratio.is_dimensionless()
