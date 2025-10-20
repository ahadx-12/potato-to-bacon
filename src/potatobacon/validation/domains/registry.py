from __future__ import annotations
from typing import Dict
from potatobacon.models import EquationDomain
from .base_guard import DomainGuard
from .classical_guard import ClassicalGuard
from .quantum_guard import QuantumGuard
from .statistical_guard import StatisticalGuard

# Stateless singletons (performance & consistency)
_DOMAIN_GUARD_REGISTRY: Dict[EquationDomain, DomainGuard] = {
    EquationDomain.CLASSICAL: ClassicalGuard(),
    EquationDomain.RELATIVISTIC: ClassicalGuard(),  # reuse until a dedicated relativity guard exists
    EquationDomain.QUANTUM: QuantumGuard(),
    EquationDomain.STATISTICAL: StatisticalGuard(),
}

def get_guard_for_domain(domain: EquationDomain) -> DomainGuard:
    guard = _DOMAIN_GUARD_REGISTRY.get(domain)
    if guard is None:
        raise ValueError(f"No domain guard registered for {domain!r}")
    return guard
