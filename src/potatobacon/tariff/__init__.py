"""CALE-TARIFF domain package.

This module promotes the tariff experiment into a first-class domain that
shares the CALE engine primitives (PolicyAtoms, solver_z3, ArbitrageHunter).
"""
from .atoms_hts import DUTY_RATES, tariff_policy_atoms
from .engine import run_tariff_hack, compute_duty_rate, apply_mutations
from .models import TariffScenario, TariffHuntRequestModel, TariffDossierModel

__all__ = [
    "DUTY_RATES",
    "tariff_policy_atoms",
    "run_tariff_hack",
    "compute_duty_rate",
    "apply_mutations",
    "TariffScenario",
    "TariffHuntRequestModel",
    "TariffDossierModel",
]
