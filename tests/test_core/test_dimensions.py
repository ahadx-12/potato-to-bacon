"""Tests for dimensional analysis primitives."""

import pytest

from potatobacon.core.dimensions import (
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


def test_dimension_creation():
    length = Dimension(length=1)
    assert length.length == 1
    assert not length.is_dimensionless()


def test_dimension_multiplication():
    result = LENGTH * TIME
    assert result.length == 1
    assert result.time == 1


def test_dimension_division():
    result = LENGTH / TIME
    assert result == VELOCITY


def test_dimension_power():
    area = LENGTH**2
    assert area.length == 2
    assert area.mass == 0


def test_velocity_construction():
    velocity = LENGTH / TIME
    assert velocity.length == 1
    assert velocity.time == -1


def test_force_construction():
    accel = LENGTH / (TIME**2)
    force = MASS * accel
    assert force == FORCE


def test_energy_construction():
    energy = FORCE * LENGTH
    assert energy == ENERGY


def test_dimensionless_check():
    assert DIMENSIONLESS.is_dimensionless()
    assert not LENGTH.is_dimensionless()
    assert (LENGTH / LENGTH).is_dimensionless()


def test_dimension_string_repr():
    assert str(DIMENSIONLESS) == "dimensionless"
    velocity_str = str(VELOCITY)
    assert "L" in velocity_str
    assert "T" in velocity_str


def test_invalid_dimension_exponent():
    with pytest.raises(ValueError):
        Dimension(length=1.5)  # type: ignore[arg-type]


def test_dimension_immutability():
    dim = Dimension(length=1)
    with pytest.raises(AttributeError):
        dim.length = 2  # type: ignore[misc]


def test_predefined_dimensions():
    assert ACCELERATION.length == 1
    assert ACCELERATION.time == -2
    assert MOMENTUM.mass == 1
    assert MOMENTUM.time == -1
    assert POWER.time == -3
