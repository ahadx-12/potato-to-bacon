"""CALE-TARIFF domain package.

This module promotes the tariff experiment into a first-class domain that
shares the CALE engine primitives (PolicyAtoms, solver_z3, ArbitrageHunter).
"""
from .atoms_hts import DUTY_RATES, tariff_policy_atoms
from .context_loader import get_default_tariff_context, get_tariff_atoms_for_context
from .engine import apply_mutations, compute_duty_rate, explain_tariff_scenario, run_tariff_hack
from .models import TariffDossierModel, TariffExplainResponseModel, TariffHuntRequestModel, TariffScenario

__all__ = [
    "DUTY_RATES",
    "tariff_policy_atoms",
    "run_tariff_hack",
    "compute_duty_rate",
    "apply_mutations",
    "explain_tariff_scenario",
    "get_default_tariff_context",
    "get_tariff_atoms_for_context",
    "TariffScenario",
    "TariffHuntRequestModel",
    "TariffDossierModel",
    "TariffExplainResponseModel",
]
