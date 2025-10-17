"""Lightweight dimensional algebra and unit expression parsing."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Sequence, Tuple
import math
import re


BASE_ORDER = ("L", "M", "T", "I", "Θ", "N", "J")
BASE_SYMBOLS = ("m", "kg", "s", "A", "K", "mol", "cd")
DIMS = Tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]


class UnitParseError(ValueError):
    """Raised when a unit expression cannot be parsed."""

    def __init__(self, message: str, text: str, position: int | None = None) -> None:
        pointer = ""
        if position is not None and 0 <= position <= len(text):
            pointer = f"\n{text}\n{' ' * position}^"
        super().__init__(f"{message}{pointer}")
        self.text = text
        self.position = position


def _to_fraction(value: float | int | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(10_000)
    try:
        return Fraction(value)
    except ValueError:
        return Fraction(float(value)).limit_denominator(10_000)


def _zero_dims() -> DIMS:
    return (Fraction(0),) * 7


@dataclass(frozen=True)
class Quantity:
    """SI-canonical quantity representation."""

    scale: float
    dims: DIMS

    def __post_init__(self) -> None:
        normalized = tuple(_to_fraction(d) for d in self.dims)
        object.__setattr__(self, "dims", normalized)

    def __mul__(self, other: "Quantity") -> "Quantity":
        return Quantity(self.scale * other.scale, _add_dims(self.dims, other.dims))

    def __truediv__(self, other: "Quantity") -> "Quantity":
        return Quantity(self.scale / other.scale, _sub_dims(self.dims, other.dims))

    def __pow__(self, exponent: float | int | Fraction) -> "Quantity":
        frac = _to_fraction(exponent)
        return Quantity(self.scale ** float(frac), _mul_dims(self.dims, frac))

    def is_dimensionless(self) -> bool:
        return all(d == 0 for d in self.dims)


def _add_dims(a: DIMS, b: DIMS) -> DIMS:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _sub_dims(a: DIMS, b: DIMS) -> DIMS:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _mul_dims(a: DIMS, multiplier: Fraction) -> DIMS:
    return tuple(x * multiplier for x in a)  # type: ignore[return-value]


class UnitRegistry:
    """Registry of known units and helper for canonical formatting."""

    def __init__(self) -> None:
        self._units: Dict[str, Quantity] = {}
        self._preferred_symbols: Dict[Tuple[Fraction, ...], str] = {}
        self._prefixes: Dict[str, float] = {
            "Y": 1e24,
            "Z": 1e21,
            "E": 1e18,
            "P": 1e15,
            "T": 1e12,
            "G": 1e9,
            "M": 1e6,
            "k": 1e3,
            "h": 1e2,
            "da": 1e1,
            "d": 1e-1,
            "c": 1e-2,
            "m": 1e-3,
            "µ": 1e-6,
            "u": 1e-6,
            "n": 1e-9,
            "p": 1e-12,
            "f": 1e-15,
            "a": 1e-18,
            "z": 1e-21,
            "y": 1e-24,
        }
        self._sorted_prefixes = sorted(self._prefixes.keys(), key=len, reverse=True)

        self._install_defaults()

    # ------------------------------------------------------------------
    def register(
        self,
        symbol: str,
        scale: float,
        dims: Sequence[int | Fraction],
        *,
        aliases: Sequence[str] | None = None,
        prefer: bool = False,
    ) -> None:
        quantity = Quantity(scale, tuple(_to_fraction(d) for d in dims))
        for key in (symbol, *(aliases or [])):
            self._units[key] = quantity
        if quantity.scale == 1.0 or prefer:
            current = self._preferred_symbols.get(quantity.dims)
            if current is None or prefer:
                self._preferred_symbols[quantity.dims] = symbol

    def get(self, symbol: str) -> Quantity:
        if symbol in self._units:
            return self._units[symbol]

        for prefix in self._sorted_prefixes:
            if symbol.startswith(prefix) and len(symbol) > len(prefix):
                tail = symbol[len(prefix) :]
                if tail in self._units:
                    return Quantity(
                        self._prefixes[prefix] * self._units[tail].scale,
                        self._units[tail].dims,
                    )
        raise UnitParseError(f"Unknown unit symbol '{symbol}'", symbol, 0)

    def prefer_symbol(self, quantity: Quantity) -> str | None:
        if abs(quantity.scale - 1.0) > 1e-12:
            return None
        return self._preferred_symbols.get(tuple(quantity.dims))

    # ------------------------------------------------------------------
    def _install_defaults(self) -> None:
        L = (1, 0, 0, 0, 0, 0, 0)
        M = (0, 1, 0, 0, 0, 0, 0)
        T = (0, 0, 1, 0, 0, 0, 0)
        I = (0, 0, 0, 1, 0, 0, 0)
        Th = (0, 0, 0, 0, 1, 0, 0)
        N = (0, 0, 0, 0, 0, 1, 0)
        J = (0, 0, 0, 0, 0, 0, 1)

        self.register("m", 1.0, L, aliases=["meter", "metre"])
        self.register("kg", 1.0, M, aliases=["kilogram"])
        self.register("g", 1e-3, M, aliases=["gram"])
        self.register("s", 1.0, T, aliases=["sec", "second"])
        self.register("min", 60.0, T)
        self.register("h", 3600.0, T, aliases=["hr", "hour"])
        self.register("day", 86400.0, T)
        self.register("A", 1.0, I, aliases=["amp", "ampere"])
        self.register("K", 1.0, Th)
        self.register("degC", 1.0, Th, aliases=["°C"], prefer=True)
        self.register("mol", 1.0, N)
        self.register("cd", 1.0, J)

        # Dimensionless helpers
        self.register("rad", 1.0, _zero_dims())
        self.register("sr", 1.0, _zero_dims())
        self.register("rev", 2 * math.pi, _zero_dims())
        self.register("deg", math.pi / 180.0, _zero_dims())

        # Derived SI
        self.register("Hz", 1.0, (0, 0, -1, 0, 0, 0, 0))
        self.register("N", 1.0, (1, 1, -2, 0, 0, 0, 0), prefer=True)
        self.register("Pa", 1.0, (-1, 1, -2, 0, 0, 0, 0), prefer=True)
        self.register("J", 1.0, (2, 1, -2, 0, 0, 0, 0), prefer=True)
        self.register("W", 1.0, (2, 1, -3, 0, 0, 0, 0), prefer=True)
        self.register("C", 1.0, (0, 0, 1, 1, 0, 0, 0), prefer=True)
        self.register("V", 1.0, (2, 1, -3, -1, 0, 0, 0), prefer=True)
        self.register("Ω", 1.0, (2, 1, -3, -2, 0, 0, 0), aliases=["ohm"], prefer=True)
        self.register("S", 1.0, (-2, -1, 3, 2, 0, 0, 0), prefer=True)
        self.register("F", 1.0, (-2, -1, 4, 2, 0, 0, 0), prefer=True)
        self.register("Wb", 1.0, (0, 1, -2, -1, 0, 0, 0), prefer=True)
        self.register("T", 1.0, (0, 1, -2, -1, 0, 0, 0))
        self.register("H", 1.0, (2, 1, -2, -2, 0, 0, 0), prefer=True)
        self.register("lx", 1.0, (-2, 0, 0, 0, 0, 0, 1), prefer=True)
        self.register("lm", 1.0, (0, 0, 0, 0, 0, 0, 1))
        self.register("kat", 1.0, (0, 0, -1, 0, 0, 1, 0))

        # Other useful units
        self.register("eV", 1.602176634e-19, (2, 1, -2, 0, 0, 0, 0))
        self.register("cal", 4.184, (2, 1, -2, 0, 0, 0, 0))
        self.register("kcal", 4184.0, (2, 1, -2, 0, 0, 0, 0))
        self.register("L", 1e-3, (3, 0, 0, 0, 0, 0, 0), aliases=["l", "liter", "litre"])
        self.register("bar", 1e5, (-1, 1, -2, 0, 0, 0, 0))
        self.register("atm", 101325.0, (-1, 1, -2, 0, 0, 0, 0))
        self.register("psi", 6894.757293168, (-1, 1, -2, 0, 0, 0, 0))
        self.register("Gauss", 1e-4, (0, 1, -2, -1, 0, 0, 0))
        self.register("G", 6.67430e-11, (-2, -1, -2, 0, 0, 0, 0))

        # Imperial / CGS helpers
        self.register("inch", 0.0254, L)
        self.register("ft", 0.3048, L)
        self.register("yd", 0.9144, L)
        self.register("mile", 1609.344, L)
        self.register("lb", 0.45359237, M, aliases=["pound"])
        self.register("slug", 14.59390294, M)
        self.register("oz", 0.028349523125, M)
        self.register("ton", 907.18474, M)
        self.register("lbf", 4.4482216152605, (1, 1, -2, 0, 0, 0, 0))
        self.register("dyn", 1e-5, (1, 1, -2, 0, 0, 0, 0))
        self.register("erg", 1e-7, (2, 1, -2, 0, 0, 0, 0))

        # Natural constants (for presets)
        self.register("c", 299792458.0, (1, 0, -1, 0, 0, 0, 0))
        self.register("ħ", 1.054571817e-34, (2, 1, -1, 0, 0, 0, 0), aliases=["hbar"], prefer=True)
        self.register("h", 6.62607015e-34, (2, 1, -1, 0, 0, 0, 0), aliases=["planck"])
        self.register("k_B", 1.380649e-23, (2, 1, -2, 0, -1, 0, 0), aliases=["kB"])
        self.register("e", 1.602176634e-19, (0, 0, 1, 1, 0, 0, 0), aliases=["q_e"])


DEFAULT_REGISTRY = UnitRegistry()
DIMENSIONLESS = Quantity(1.0, _zero_dims())


class _Token(Tuple[str, str, int, int]):
    __slots__ = ()


_SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
})


_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
    |(?P<lpar>\()
    |(?P<rpar>\))
    |(?P<op>[*/])
    |(?P<pow>\^)
    |(?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    |(?P<sym>[A-Za-zµμΩ°_][A-Za-z0-9µμΩ°_\-]*)
    """,
    re.VERBOSE,
)


def _normalize_input(text: str) -> str:
    text = text.replace("·", "*").replace("×", "*")

    def replace_superscripts(match: re.Match[str]) -> str:
        base = match.group(1)
        supers = match.group(2).translate(_SUPERSCRIPT_TRANS)
        return f"{base}^{supers}"

    text = re.sub(r"([A-Za-zµμΩ°0-9\)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)", replace_superscripts, text)

    def replace_implicit(match: re.Match[str]) -> str:
        sym = match.group(1)
        exp = match.group(2)
        return f"{sym}^{exp}"

    text = re.sub(r"(?<![A-Za-zµμΩ°0-9_])([A-Za-zµμΩ°]+)([-+]?\d+)", replace_implicit, text)
    return text


def _tokenize(text: str) -> List[_Token]:
    tokens: List[_Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            raise UnitParseError(
                f"Unexpected character '{text[pos]}' in unit expression", text, pos
            )
        if match.lastgroup != "space":
            tokens.append(_Token((match.lastgroup, match.group(0), match.start(), match.end())))
        pos = match.end()
    return tokens


class _TokenStream:
    def __init__(self, tokens: List[_Token], original: str) -> None:
        self.tokens = tokens
        self.original = original
        self.index = 0

    def _skip_spaces(self) -> None:
        pass  # spaces already removed

    def peek(self) -> _Token | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def pop(self, expected: str | None = None) -> _Token:
        token = self.peek()
        if token is None:
            raise UnitParseError("Unexpected end of unit expression", self.original, len(self.original))
        kind, value, start, _ = token
        if expected and kind != expected:
            raise UnitParseError(
                f"Expected {expected} but found '{value}'", self.original, start
            )
        self.index += 1
        return token


def parse_unit_expr(text: str, *, registry: UnitRegistry | None = None) -> Quantity:
    """Parse ``text`` into a :class:`Quantity` canonicalised to SI units."""

    if registry is None:
        registry = DEFAULT_REGISTRY

    stripped = text.strip()
    if not stripped:
        raise UnitParseError("Unit expression is empty", text, 0)

    normalized = _normalize_input(stripped)
    tokens = _tokenize(normalized)
    stream = _TokenStream(tokens, normalized)

    quantity = _parse_expr(stream, registry)
    if stream.peek() is not None:
        kind, value, start, _ = stream.peek()  # type: ignore[misc]
        raise UnitParseError(f"Unexpected token '{value}'", normalized, start)
    return quantity


def _parse_expr(stream: _TokenStream, registry: UnitRegistry) -> Quantity:
    q = _parse_term(stream, registry)
    while True:
        next_tok = stream.peek()
        if next_tok and next_tok[0] == "op" and next_tok[1] == "/":
            stream.pop("op")
            q = q / _parse_term(stream, registry)
        else:
            break
    return q


def _parse_term(stream: _TokenStream, registry: UnitRegistry) -> Quantity:
    q = _parse_factor(stream, registry)
    while True:
        next_tok = stream.peek()
        if not next_tok:
            break
        kind, value, _, _ = next_tok
        if kind == "op" and value == "*":
            stream.pop("op")
            q = q * _parse_factor(stream, registry)
            continue
        if kind in {"sym", "num", "lpar"}:
            q = q * _parse_factor(stream, registry)
            continue
        break
    return q


def _parse_factor(stream: _TokenStream, registry: UnitRegistry) -> Quantity:
    q = _parse_atom(stream, registry)
    next_tok = stream.peek()
    if next_tok and next_tok[0] == "pow":
        stream.pop("pow")
        exp_tok = stream.pop("num")
        exponent = _to_fraction(exp_tok[1])
        q = q ** exponent
    return q


def _parse_atom(stream: _TokenStream, registry: UnitRegistry) -> Quantity:
    token = stream.peek()
    if token is None:
        raise UnitParseError("Unexpected end of unit expression", stream.original, len(stream.original))
    kind, value, start, _ = token
    if kind == "lpar":
        stream.pop("lpar")
        inner = _parse_expr(stream, registry)
        stream.pop("rpar")
        return inner
    if kind == "sym":
        stream.pop("sym")
        return registry.get(value)
    if kind == "num":
        stream.pop("num")
        return Quantity(float(value), _zero_dims())
    raise UnitParseError(f"Unexpected token '{value}'", stream.original, start)


def format_quantity(quantity: Quantity, *, registry: UnitRegistry | None = None) -> str:
    """Return a human readable canonical string for ``quantity``."""

    registry = registry or DEFAULT_REGISTRY

    preferred = registry.prefer_symbol(quantity)
    if preferred:
        return preferred

    numerator: List[str] = []
    denominator: List[str] = []
    for symbol, exponent in zip(BASE_SYMBOLS, quantity.dims):
        if exponent == 0:
            continue
        target = numerator if exponent > 0 else denominator
        abs_exp = -exponent if exponent < 0 else exponent
        formatted_exp = _format_exponent(abs_exp)
        if formatted_exp == "1":
            target.append(symbol)
        else:
            target.append(f"{symbol}^{formatted_exp}")

    if numerator:
        unit_str = "*".join(numerator)
    else:
        unit_str = "1"

    if denominator:
        unit_str = f"{unit_str}/" + "*".join(denominator)

    scale = quantity.scale
    if math.isclose(scale, 1.0, rel_tol=0, abs_tol=1e-12):
        return unit_str
    return f"{_format_scale(scale)}*{unit_str}" if unit_str != "1" else _format_scale(scale)


def _format_scale(scale: float) -> str:
    if scale == 0:
        return "0"
    if 1e-3 < abs(scale) < 1e4:
        return f"{scale:g}"
    formatted = f"{scale:.6e}"
    return formatted.rstrip("0").rstrip(".")


def _format_exponent(exponent: Fraction) -> str:
    exponent = exponent.limit_denominator()
    if exponent.denominator == 1:
        return str(exponent.numerator)
    return f"{exponent.numerator}/{exponent.denominator}"


def dims_to_vector(dims: DIMS) -> Tuple[Fraction, ...]:
    return tuple(dims)


def dims_to_pretty(dims: DIMS) -> str:
    parts = []
    for axis, value in zip(BASE_ORDER, dims):
        if value == 0:
            continue
        exponent = _format_exponent(abs(value))
        if value > 0:
            parts.append(f"{axis}^{exponent}")
        else:
            parts.append(f"{axis}^-{exponent}")
    return "1" if not parts else " · ".join(parts)


__all__ = [
    "Quantity",
    "UnitParseError",
    "UnitRegistry",
    "DEFAULT_REGISTRY",
    "DIMENSIONLESS",
    "BASE_ORDER",
    "BASE_SYMBOLS",
    "parse_unit_expr",
    "format_quantity",
    "dims_to_pretty",
]

