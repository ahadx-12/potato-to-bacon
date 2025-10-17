"""High-level helpers for unit presets, suggestions, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from potatobacon.units.algebra import DEFAULT_REGISTRY, Quantity, UnitParseError, format_quantity, parse_unit_expr


UNIT_SYSTEMS: Dict[str, Dict[str, str]] = {
    "SI": {
        "length": "m",
        "area": "m^2",
        "volume": "m^3",
        "mass": "kg",
        "time": "s",
        "frequency": "Hz",
        "angle": "rad",
        "angular_velocity": "rad/s",
        "temperature": "K",
        "amount": "mol",
        "luminosity": "cd",
        "velocity": "m/s",
        "acceleration": "m/s^2",
        "force": "N",
        "pressure": "Pa",
        "energy": "J",
        "power": "W",
        "momentum": "kg*m/s",
        "charge": "C",
        "potential": "V",
        "current": "A",
        "resistance": "Ω",
        "capacitance": "F",
        "inductance": "H",
        "magnetic_flux": "Wb",
        "magnetic_flux_density": "T",
        "density": "kg/m^3",
        "work": "J",
        "energy_density": "J/m^3",
        "dimensionless": "1",
    },
    "CGS": {
        "length": "cm",
        "area": "cm^2",
        "volume": "cm^3",
        "mass": "g",
        "time": "s",
        "frequency": "Hz",
        "velocity": "cm/s",
        "acceleration": "cm/s^2",
        "force": "dyn",
        "pressure": "dyn/cm^2",
        "energy": "erg",
        "power": "erg/s",
        "momentum": "g*cm/s",
        "density": "g/cm^3",
        "temperature": "K",
        "charge": "C",
        "dimensionless": "1",
    },
    "Imperial": {
        "length": "ft",
        "area": "ft^2",
        "volume": "ft^3",
        "mass": "lb",
        "time": "s",
        "velocity": "ft/s",
        "acceleration": "ft/s^2",
        "force": "lbf",
        "pressure": "psi",
        "energy": "ft*lbf",
        "power": "ft*lbf/s",
        "momentum": "lb*ft/s",
        "density": "slug/ft^3",
        "temperature": "K",
        "dimensionless": "1",
    },
    "Natural": {
        "velocity": "c",
        "mass": "eV/c^2",
        "energy": "eV",
        "time": "ħ/eV",
        "length": "ħ/c",
        "charge": "e",
        "temperature": "eV/kB",
        "dimensionless": "1",
    },
}


VAR_KIND: Dict[str, str] = {
    "m": "mass",
    "mass": "mass",
    "M": "mass",
    "rho": "density",
    "density": "density",
    "v": "velocity",
    "u": "velocity",
    "speed": "velocity",
    "velocity": "velocity",
    "c": "velocity",
    "a": "acceleration",
    "g": "acceleration",
    "acc": "acceleration",
    "acceleration": "acceleration",
    "F": "force",
    "force": "force",
    "E": "energy",
    "energy": "energy",
    "KE": "energy",
    "PE": "energy",
    "P": "power",
    "power": "power",
    "p": "momentum",
    "momentum": "momentum",
    "x": "length",
    "y": "length",
    "z": "length",
    "r": "length",
    "L": "length",
    "length": "length",
    "radius": "length",
    "position": "length",
    "distance": "length",
    "s": "length",
    "A": "area",
    "area": "area",
    "V": "volume",
    "volume": "volume",
    "t": "time",
    "time": "time",
    "dt": "time",
    "omega": "angular_velocity",
    "Ω": "angular_velocity",
    "theta": "angle",
    "phi": "angle",
    "angle": "angle",
    "I": "current",
    "current": "current",
    "q": "charge",
    "Q": "charge",
    "charge": "charge",
    "Vpot": "potential",
    "voltage": "potential",
    "potential": "potential",
    "R": "resistance",
    "resistance": "resistance",
    "Ccap": "capacitance",
    "capacitance": "capacitance",
    "Lind": "inductance",
    "inductance": "inductance",
    "B": "magnetic_flux_density",
    "magnetic_flux_density": "magnetic_flux_density",
    "Phi": "magnetic_flux",
    "magnetic_flux": "magnetic_flux",
    "Ttemp": "temperature",
    "temp": "temperature",
    "temperature": "temperature",
    "n": "amount",
    "amount": "amount",
    "N": "force",
    "frequency": "frequency",
    "f": "frequency",
    "mu": "dimensionless",
    "dimensionless": "dimensionless",
}


@dataclass(slots=True)
class UnitDiagnostic:
    """Structured diagnostic returned when unit parsing fails."""

    symbol: str
    code: str
    message: str
    hint: str | None = None


def detect_kind(symbol: str) -> str | None:
    """Guess the semantic kind of ``symbol`` using heuristics."""

    clean = symbol.strip()
    if not clean:
        return None
    if clean in VAR_KIND:
        return VAR_KIND[clean]
    lower = clean.lower()
    if lower in VAR_KIND:
        return VAR_KIND[lower]
    return None


def apply_unit_system(variables: Iterable[str], system: str = "SI") -> Dict[str, str]:
    """Return suggested units for ``variables`` using ``system`` presets."""

    if system not in UNIT_SYSTEMS:
        raise ValueError(f"Unknown unit system '{system}'")
    sys = UNIT_SYSTEMS[system]
    suggestions: Dict[str, str] = {}
    for symbol in variables:
        kind = detect_kind(symbol)
        if kind and kind in sys:
            suggestions[symbol] = sys[kind]
        else:
            suggestions[symbol] = "?"
    return suggestions


def suggest_units(symbols: Iterable[str], *, system: str = "SI", existing: Dict[str, str] | None = None) -> Dict[str, str]:
    """Suggest units for ``symbols`` using heuristics and presets.

    ``existing`` units are preserved and override the suggestions.
    """

    existing = existing or {}
    sys = UNIT_SYSTEMS.get(system, UNIT_SYSTEMS["SI"])
    suggestions: Dict[str, str] = {}
    for symbol in symbols:
        if symbol in existing and existing[symbol]:
            suggestions[symbol] = existing[symbol]
            continue
        kind = detect_kind(symbol)
        unit = sys.get(kind or "", "?") if kind else "?"
        if unit != "?":
            suggestions[symbol] = unit
    return suggestions


def analyze_units_map(
    units: Dict[str, str],
    *,
    registry=DEFAULT_REGISTRY,
) -> Tuple[Dict[str, Quantity], Dict[str, str], List[UnitDiagnostic]]:
    """Parse ``units`` into quantities, returning diagnostics if needed."""

    quantities: Dict[str, Quantity] = {}
    canonical: Dict[str, str] = {}
    diagnostics: List[UnitDiagnostic] = []

    for symbol, unit_text in sorted(units.items()):
        text = (unit_text or "").strip()
        if not text:
            diagnostics.append(
                UnitDiagnostic(
                    symbol=symbol,
                    code="missing",
                    message="Unit is missing",
                    hint="Provide a unit for this symbol or remove it from the list.",
                )
            )
            continue
        try:
            quantity = parse_unit_expr(text, registry=registry)
        except UnitParseError as exc:
            diagnostics.append(
                UnitDiagnostic(
                    symbol=symbol,
                    code="parse-error",
                    message=str(exc),
                    hint="Check for typos or mismatched parentheses in the unit expression.",
                )
            )
            continue

        quantities[symbol] = quantity
        canonical[symbol] = format_quantity(quantity, registry=registry)

    return quantities, canonical, diagnostics


__all__ = [
    "UNIT_SYSTEMS",
    "VAR_KIND",
    "UnitDiagnostic",
    "detect_kind",
    "apply_unit_system",
    "suggest_units",
    "analyze_units_map",
]

