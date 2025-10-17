"""Physical quantity utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import sympy as sp

from .dimensions import Dimension, DimensionalError


@dataclass
class PhysicalQuantity:
    """A symbolic quantity paired with dimensions and optional metadata."""

    symbol: str
    expression: sp.Expr
    dimensions: Dimension
    unit: Optional[str] = None
    domain: str = "classical"
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("Symbol must be a non-empty string")

        if not isinstance(self.expression, sp.Expr):
            try:
                self.expression = sp.sympify(self.expression)
            except Exception as exc:  # pragma: no cover - defensive
                raise TypeError("Expression must be convertible to SymPy") from exc

        if not isinstance(self.dimensions, Dimension):
            raise TypeError("Dimensions must be a Dimension instance")

    def __mul__(self, other: PhysicalQuantity) -> PhysicalQuantity:
        if not isinstance(other, PhysicalQuantity):
            raise TypeError(f"Cannot multiply PhysicalQuantity by {type(other)}")

        return PhysicalQuantity(
            symbol=f"({self.symbol}*{other.symbol})",
            expression=self.expression * other.expression,
            dimensions=self.dimensions * other.dimensions,
            domain=self._merge_domain(other),
        )

    def __truediv__(self, other: PhysicalQuantity) -> PhysicalQuantity:
        if not isinstance(other, PhysicalQuantity):
            raise TypeError(f"Cannot divide PhysicalQuantity by {type(other)}")

        return PhysicalQuantity(
            symbol=f"({self.symbol}/{other.symbol})",
            expression=self.expression / other.expression,
            dimensions=self.dimensions / other.dimensions,
            domain=self._merge_domain(other),
        )

    def __pow__(self, exponent: int) -> PhysicalQuantity:
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")

        return PhysicalQuantity(
            symbol=f"{self.symbol}^{exponent}",
            expression=self.expression**exponent,
            dimensions=self.dimensions**exponent,
            domain=self.domain,
        )

    def __add__(self, other: PhysicalQuantity) -> PhysicalQuantity:
        if not isinstance(other, PhysicalQuantity):
            raise TypeError(f"Cannot add PhysicalQuantity and {type(other)}")
        if self.dimensions != other.dimensions:
            raise DimensionalError(
                "Cannot add quantities with different dimensions: "
                f"{self.dimensions} vs {other.dimensions}"
            )

        return PhysicalQuantity(
            symbol=f"({self.symbol}+{other.symbol})",
            expression=self.expression + other.expression,
            dimensions=self.dimensions,
            domain=self._merge_domain(other),
        )

    def __sub__(self, other: PhysicalQuantity) -> PhysicalQuantity:
        if not isinstance(other, PhysicalQuantity):
            raise TypeError(f"Cannot subtract {type(other)} from PhysicalQuantity")
        if self.dimensions != other.dimensions:
            raise DimensionalError(
                "Cannot subtract quantities with different dimensions: "
                f"{self.dimensions} vs {other.dimensions}"
            )

        return PhysicalQuantity(
            symbol=f"({self.symbol}-{other.symbol})",
            expression=self.expression - other.expression,
            dimensions=self.dimensions,
            domain=self._merge_domain(other),
        )

    def _merge_domain(self, other: PhysicalQuantity) -> str:
        hierarchy = ["classical", "relativistic", "quantum"]
        try:
            idx_self = hierarchy.index(self.domain)
        except ValueError:
            idx_self = 0
        try:
            idx_other = hierarchy.index(other.domain)
        except ValueError:
            idx_other = 0
        return hierarchy[max(idx_self, idx_other)]

    def is_dimensionless(self) -> bool:
        return self.dimensions.is_dimensionless()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            "PhysicalQuantity("
            f"symbol={self.symbol}, dimensions={self.dimensions}, domain={self.domain})"
        )
