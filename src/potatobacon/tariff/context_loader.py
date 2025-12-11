from __future__ import annotations

"""Context-aware loader for tariff policy atoms."""

from typing import Callable, Dict, List

from potatobacon.law.solver_z3 import PolicyAtom

from .atoms_hts import tariff_policy_atoms


DEFAULT_TARIFF_CONTEXT = "HTS_US_2025_Q1"

_CONTEXT_LOADERS: Dict[str, Callable[[], List[PolicyAtom]]] = {
    DEFAULT_TARIFF_CONTEXT: tariff_policy_atoms,
}


def get_default_tariff_context() -> str:
    """Return the default tariff law context identifier."""

    return DEFAULT_TARIFF_CONTEXT


def get_tariff_atoms_for_context(context: str | None) -> List[PolicyAtom]:
    """Return tariff policy atoms for the requested context.

    Raises a ``ValueError`` if the context is unknown to make versioning explicit.
    """

    resolved = context or DEFAULT_TARIFF_CONTEXT
    if resolved not in _CONTEXT_LOADERS:
        raise ValueError(f"Unknown tariff context: {resolved}")
    return _CONTEXT_LOADERS[resolved]()
