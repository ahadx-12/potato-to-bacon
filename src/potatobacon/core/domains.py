"""Definitions and metadata for physics domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DomainMetadata:
    name: str
    description: str
    typical_scales: Dict[str, str]
    assumptions: List[str]
    validation_rules: List[str]
    examples: List[str]


CLASSICAL = DomainMetadata(
    name="classical",
    description="Newtonian mechanics for non-relativistic systems",
    typical_scales={
        "length": "mm to km",
        "velocity": "v << c",
        "energy": "eV to MJ",
        "time": "microseconds to years",
    },
    assumptions=[
        "Galilean relativity",
        "Absolute time",
        "Deterministic evolution",
        "Negligible quantum effects",
    ],
    validation_rules=[
        "dimensional_consistency",
        "constraint_satisfaction",
        "energy_conservation",
    ],
    examples=[
        "F = ma",
        "KE = (1/2)mv^2",
        "U = mgh",
    ],
)

RELATIVISTIC = DomainMetadata(
    name="relativistic",
    description="Special and general relativity for high velocity or strong gravity",
    typical_scales={
        "velocity": "v ~ c",
        "energy": "Rest-mass energy significant",
        "gravity": "Strong curvature",
    },
    assumptions=[
        "Lorentz invariance",
        "Speed of light constant",
        "Spacetime manifold",
        "Equivalence principle",
    ],
    validation_rules=[
        "lorentz_invariance",
        "causality",
        "v_less_than_c",
        "dimensional_consistency",
    ],
    examples=[
        "E^2 = (pc)^2 + (mc^2)^2",
        "t' = gamma * (t - vx/c^2)",
        "ds^2 = -c^2 dt^2 + dx^2",
    ],
)

QUANTUM = DomainMetadata(
    name="quantum",
    description="Quantum mechanics with wave functions and uncertainty",
    typical_scales={
        "length": "atomic to nanoscale",
        "energy": "eV to keV",
        "action": "~ hbar",
    },
    assumptions=[
        "Wave function formalism",
        "Superposition principle",
        "Measurement postulate",
        "Heisenberg uncertainty",
    ],
    validation_rules=[
        "hermitian_observables",
        "unitary_evolution",
        "commutation_relations",
        "dimensional_consistency",
    ],
    examples=[
        "H psi = i hbar dpsi/dt",
        "Delta x Delta p >= hbar/2",
        "[x, p] = i hbar",
    ],
)

STATISTICAL = DomainMetadata(
    name="statistical",
    description="Statistical mechanics and thermodynamics for many-body systems",
    typical_scales={
        "particles": "Avogadro scale",
        "temperature": "mK to thousands of K",
        "entropy": "J/K",
    },
    assumptions=[
        "Ergodic hypothesis",
        "Thermodynamic limit",
        "Appropriate statistics (MB/FD/BE)",
    ],
    validation_rules=[
        "second_law_thermodynamics",
        "partition_function_normalization",
        "dimensional_consistency",
    ],
    examples=[
        "S = k_B ln(Ω)",
        "F = U - TS",
        "Z = Σ exp(-E_i/k_B T)",
    ],
)

DOMAIN_REGISTRY: Dict[str, DomainMetadata] = {
    "classical": CLASSICAL,
    "relativistic": RELATIVISTIC,
    "quantum": QUANTUM,
    "statistical": STATISTICAL,
}


def get_domain(name: str) -> Optional[DomainMetadata]:
    return DOMAIN_REGISTRY.get(name.lower())


def list_domains() -> List[str]:
    return list(DOMAIN_REGISTRY.keys())
