"""Dimensional analysis primitives for the potato.to.bacon project."""

from __future__ import annotations

from dataclasses import dataclass


class DimensionalError(Exception):
    """Raised when a dimensional operation is invalid."""


@dataclass(frozen=True)
class Dimension:
    """Representation of physical dimensions as integer exponents.

    The seven base SI units are tracked: length (L), mass (M), time (T),
    electric current (I), temperature (Θ), amount of substance (N) and
    luminous intensity (J).
    """

    length: int = 0
    mass: int = 0
    time: int = 0
    current: int = 0
    temperature: int = 0
    amount: int = 0
    luminosity: int = 0

    def __post_init__(self) -> None:
        for field in (
            "length",
            "mass",
            "time",
            "current",
            "temperature",
            "amount",
            "luminosity",
        ):
            value = getattr(self, field)
            if not isinstance(value, int):
                raise ValueError(
                    f"Dimension exponent {field} must be an integer, got {type(value)}"
                )

    def __mul__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            raise TypeError(f"Cannot multiply Dimension by {type(other)}")

        return Dimension(
            length=self.length + other.length,
            mass=self.mass + other.mass,
            time=self.time + other.time,
            current=self.current + other.current,
            temperature=self.temperature + other.temperature,
            amount=self.amount + other.amount,
            luminosity=self.luminosity + other.luminosity,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        if not isinstance(other, Dimension):
            raise TypeError(f"Cannot divide Dimension by {type(other)}")

        return Dimension(
            length=self.length - other.length,
            mass=self.mass - other.mass,
            time=self.time - other.time,
            current=self.current - other.current,
            temperature=self.temperature - other.temperature,
            amount=self.amount - other.amount,
            luminosity=self.luminosity - other.luminosity,
        )

    def __pow__(self, exponent: int) -> Dimension:
        if not isinstance(exponent, int):
            raise TypeError(f"Dimension exponent must be integer, got {type(exponent)}")

        return Dimension(
            length=self.length * exponent,
            mass=self.mass * exponent,
            time=self.time * exponent,
            current=self.current * exponent,
            temperature=self.temperature * exponent,
            amount=self.amount * exponent,
            luminosity=self.luminosity * exponent,
        )

    def is_dimensionless(self) -> bool:
        return all(
            value == 0
            for value in (
                self.length,
                self.mass,
                self.time,
                self.current,
                self.temperature,
                self.amount,
                self.luminosity,
            )
        )

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        if self.is_dimensionless():
            return "dimensionless"

        parts = []
        mapping = {
            "L": self.length,
            "M": self.mass,
            "T": self.time,
            "I": self.current,
            "Θ": self.temperature,
            "N": self.amount,
            "J": self.luminosity,
        }

        for symbol, power in mapping.items():
            if power == 0:
                continue
            if power == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}^{power}")

        return " * ".join(parts) if parts else "dimensionless"


DIMENSIONLESS = Dimension()
LENGTH = Dimension(length=1)
MASS = Dimension(mass=1)
TIME = Dimension(time=1)
VELOCITY = Dimension(length=1, time=-1)
ACCELERATION = Dimension(length=1, time=-2)
FORCE = Dimension(mass=1, length=1, time=-2)
ENERGY = Dimension(mass=1, length=2, time=-2)
POWER = Dimension(mass=1, length=2, time=-3)
MOMENTUM = Dimension(mass=1, length=1, time=-1)
ANGULAR_MOMENTUM = Dimension(mass=1, length=2, time=-1)
