"""Dimensional analysis primitives for the potato.to.bacon project.

This module models physical dimensions as integer exponent vectors over the
seven SI base quantities (mass, length, time, electric current, temperature,
amount of substance, luminous intensity). The :class:`Dimension` type provides
closed operations under multiplication, division and integer exponentiation,
which mirror the algebra of dimensional analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


class DimensionalError(Exception):
    """Raised when a dimensional operation is invalid."""


_BASE_FIELDS: Tuple[str, ...] = (
    "mass",
    "length",
    "time",
    "current",
    "temperature",
    "amount",
    "luminosity",
)


@dataclass(frozen=True)
class Dimension:
    """Representation of physical dimensions using integer exponents.

    Each instance stores the exponent of the corresponding SI base unit in the
    order ``(M, L, T, I, Θ, N, J)``. The class enforces integer exponents and
    exposes arithmetic operations that follow the vector algebra of dimensions.
    """

    mass: int = 0
    length: int = 0
    time: int = 0
    current: int = 0
    temperature: int = 0
    amount: int = 0
    luminosity: int = 0

    def __post_init__(self) -> None:
        """Ensure all exponents are integers."""
        for field in _BASE_FIELDS:
            value = getattr(self, field)
            if not isinstance(value, int):
                raise ValueError(
                    f"Dimension exponent {field} must be an integer, got {type(value)}"
                )

    # -- Core algebra -----------------------------------------------------
    def __mul__(self, other: Dimension) -> Dimension:
        """Multiply two dimensions by adding their exponent vectors."""
        if not isinstance(other, Dimension):
            raise TypeError(f"Cannot multiply Dimension by {type(other)}")

        return Dimension(*[a + b for a, b in zip(self._as_tuple(), other._as_tuple())])

    def __truediv__(self, other: Dimension) -> Dimension:
        """Divide two dimensions by subtracting exponent vectors."""
        if not isinstance(other, Dimension):
            raise TypeError(f"Cannot divide Dimension by {type(other)}")

        return Dimension(*[a - b for a, b in zip(self._as_tuple(), other._as_tuple())])

    def __pow__(self, exponent: int) -> Dimension:
        """Raise the dimension to an integer power."""
        if not isinstance(exponent, int):
            raise TypeError(f"Dimension exponent must be integer, got {type(exponent)}")

        return Dimension(*[value * exponent for value in self._as_tuple()])

    # -- Helpers ----------------------------------------------------------
    def _as_tuple(self) -> Tuple[int, ...]:
        return tuple(getattr(self, field) for field in _BASE_FIELDS)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return self._as_tuple() == other._as_tuple()

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self._as_tuple())

    def is_dimensionless(self) -> bool:
        """Return ``True`` when all exponents are zero."""
        return all(value == 0 for value in self._as_tuple())

    @classmethod
    def dimensionless(cls) -> Dimension:
        """Construct the dimensionless unit."""
        return cls()

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        if self.is_dimensionless():
            return "dimensionless"

        parts = []
        symbols = ("M", "L", "T", "I", "Θ", "N", "J")
        for symbol, power in zip(symbols, self._as_tuple()):
            if power == 0:
                continue
            if power == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}^{power}")

        return " * ".join(parts) if parts else "dimensionless"


DIMENSIONLESS = Dimension.dimensionless()
MASS = Dimension(mass=1)
LENGTH = Dimension(length=1)
TIME = Dimension(time=1)
CURRENT = Dimension(current=1)
TEMPERATURE = Dimension(temperature=1)
AMOUNT = Dimension(amount=1)
LUMINOSITY = Dimension(luminosity=1)

VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / (TIME**2)
FORCE = MASS * ACCELERATION
ENERGY = FORCE * LENGTH
POWER = ENERGY / TIME
MOMENTUM = MASS * VELOCITY
ANGULAR_MOMENTUM = MOMENTUM * LENGTH
