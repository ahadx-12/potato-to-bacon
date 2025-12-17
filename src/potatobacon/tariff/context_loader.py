from __future__ import annotations

"""Context-aware loader for tariff policy atoms."""

from potatobacon.law.solver_z3 import PolicyAtom

from .context_registry import DEFAULT_CONTEXT_ID, load_atoms_for_context


def get_default_tariff_context() -> str:
    """Return the default tariff law context identifier."""

    return DEFAULT_CONTEXT_ID


def get_tariff_atoms_for_context(context: str | None) -> list[PolicyAtom]:
    """Return tariff policy atoms for the requested context."""

    resolved = context or DEFAULT_CONTEXT_ID
    atoms, _ = load_atoms_for_context(resolved)
    return atoms
