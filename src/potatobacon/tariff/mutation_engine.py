"""Z3-driven mutation derivation engine.

Instead of hardcoded per-category mutation lists, this engine inspects
the loaded PolicyAtoms to discover which fact changes could shift a
product into a lower-duty classification bracket.  For each candidate
target atom it computes the minimal fact patch, wraps it as a
:class:`MutationCandidate`, and returns a ranked list.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from potatobacon.law.solver_z3 import PolicyAtom, check_scenario
from potatobacon.tariff.atom_utils import duty_rate_index
from potatobacon.tariff.fact_mapper import compute_fact_gap
from potatobacon.tariff.fact_vocabulary import expand_facts
from potatobacon.tariff.models import TariffScenario
from potatobacon.tariff.mutation_generator import MutationCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derived mutation from atom analysis
# ---------------------------------------------------------------------------
@dataclass
class DerivedMutation:
    """A mutation discovered by comparing atom guard requirements."""

    target_atom: PolicyAtom
    fact_patch: Dict[str, Any]
    projected_duty_rate: float
    savings_vs_baseline: float
    human_description: str
    guard_diff: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False


# ---------------------------------------------------------------------------
# Category-level fact descriptions for human-readable summaries
# ---------------------------------------------------------------------------
_FACT_DESCRIPTIONS: Dict[str, str] = {
    "material_steel": "steel construction",
    "material_aluminum": "aluminum construction",
    "material_textile": "textile material",
    "material_rubber": "rubber material",
    "material_plastic": "plastic material",
    "material_leather": "leather material",
    "material_synthetic": "synthetic material",
    "surface_contact_textile_gt_50": "textile ground contact >50%",
    "surface_contact_rubber_gt_50": "rubber ground contact >50%",
    "felt_covering_gt_50": "felt/textile overlay >50%",
    "product_type_chassis_bolt": "chassis bolt classification",
    "product_type_electronics": "electronics classification",
    "product_type_vehicle": "vehicle classification",
    "product_type_auto_part": "auto parts classification",
    "product_type_battery": "battery classification",
    "product_type_toy": "toy classification",
    "assembled_in_usmca": "USMCA assembly qualification",
    "origin_component_qualifying": "qualifying USMCA origin components",
    "green_energy_certified": "green energy certification",
    "end_use_electric_vehicle": "electric vehicle end use",
    "textile_knit": "knit construction",
    "textile_woven": "woven construction",
    "fiber_cotton_dominant": "cotton-dominant fiber content",
    "fiber_polyester_dominant": "polyester-dominant fiber content",
    "assembly_state_kit": "knocked-down kit form",
    "housing_material_plastic": "plastic housing",
}


def _describe_fact_change(fact: str, value: Any) -> str:
    """Human-readable description for a single fact change."""
    desc = _FACT_DESCRIPTIONS.get(fact)
    if desc:
        return f"Enable {desc}" if value else f"Remove {desc}"
    action = "Set" if value else "Clear"
    return f"{action} {fact.replace('_', ' ')}"


def _describe_mutation(patch: Dict[str, Any], target: PolicyAtom) -> str:
    """Build a human-readable summary for a derived mutation."""
    parts = [_describe_fact_change(fact, val) for fact, val in sorted(patch.items())]
    target_desc = getattr(target, "text", "") or target.source_id
    if len(parts) <= 2:
        change_str = " and ".join(parts)
    else:
        change_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"{change_str} to qualify for {target_desc}"


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
class MutationEngine:
    """Discovers duty-reducing mutations by analyzing PolicyAtom guard structures.

    Given a baseline scenario and the full atom set, the engine:
    1. Identifies the active (baseline) duty atom and its rate.
    2. Finds all atoms with a *lower* duty rate.
    3. Computes the minimal fact patch to activate each cheaper atom.
    4. Verifies each patch via Z3 to confirm satisfiability.
    5. Returns ranked :class:`DerivedMutation` objects.
    """

    def __init__(
        self,
        atoms: List[PolicyAtom],
        duty_rates: Dict[str, float] | None = None,
    ) -> None:
        self.atoms = atoms
        self.duty_rates = duty_rates or duty_rate_index(atoms)

    # -- public API ---------------------------------------------------------

    def discover_mutations(
        self,
        baseline: TariffScenario,
        baseline_duty_rate: float,
        *,
        max_candidates: int = 10,
        max_patch_size: int = 5,
    ) -> List[DerivedMutation]:
        """Return ranked mutations that could lower the duty rate.

        Parameters
        ----------
        baseline : TariffScenario
            Current product scenario with compiled facts.
        baseline_duty_rate : float
            The duty rate from the baseline classification.
        max_candidates : int
            Cap on returned candidates.
        max_patch_size : int
            Skip atoms requiring more fact changes than this.
        """
        cheaper_atoms = self._find_cheaper_atoms(baseline_duty_rate)
        if not cheaper_atoms:
            return []

        candidates: List[DerivedMutation] = []
        for target_atom, target_rate in cheaper_atoms:
            patch = self._compute_fact_patch(baseline.facts, target_atom)
            if not patch or len(patch) > max_patch_size:
                continue

            # Verify via Z3 (expand facts through vocabulary bridge first)
            mutated_facts = expand_facts({**deepcopy(baseline.facts), **patch})
            is_sat, active = check_scenario(mutated_facts, self.atoms)
            if not is_sat:
                continue

            # Confirm the target atom is actually activated
            active_ids = {a.source_id for a in active}
            if target_atom.source_id not in active_ids:
                # The target wasn't activated; check if we still got a lower rate
                activated_rate = self._best_rate_from_active(active)
                if activated_rate is None or activated_rate >= baseline_duty_rate:
                    continue
                effective_rate = activated_rate
            else:
                effective_rate = target_rate

            savings = baseline_duty_rate - effective_rate
            if savings <= 0:
                continue

            candidates.append(
                DerivedMutation(
                    target_atom=target_atom,
                    fact_patch=patch,
                    projected_duty_rate=effective_rate,
                    savings_vs_baseline=savings,
                    human_description=_describe_mutation(patch, target_atom),
                    guard_diff=patch,
                    verified=True,
                )
            )

        # Rank by savings descending
        candidates.sort(key=lambda m: -m.savings_vs_baseline)
        return candidates[:max_candidates]

    def to_mutation_candidates(
        self,
        derived: List[DerivedMutation],
    ) -> List[MutationCandidate]:
        """Convert derived mutations to the existing MutationCandidate format."""
        results: List[MutationCandidate] = []
        for dm in derived:
            tradeoffs = f"Projected rate: {dm.projected_duty_rate}% (saves {dm.savings_vs_baseline}%)"
            risk = "Z3-verified satisfiable" if dm.verified else "Unverified"
            results.append(
                MutationCandidate(
                    mutation_patch=dm.fact_patch,
                    human_description=dm.human_description,
                    expected_tradeoffs=tradeoffs,
                    risk_hint=risk,
                )
            )
        return results

    # -- internals ----------------------------------------------------------

    def _find_cheaper_atoms(
        self, baseline_rate: float
    ) -> List[Tuple[PolicyAtom, float]]:
        """Return atoms with a duty rate strictly lower than baseline."""
        cheaper: List[Tuple[PolicyAtom, float]] = []
        for atom in self.atoms:
            rate = self.duty_rates.get(atom.source_id)
            if rate is not None and rate < baseline_rate:
                cheaper.append((atom, rate))
        cheaper.sort(key=lambda pair: pair[1])
        return cheaper

    def _compute_fact_patch(
        self,
        current_facts: Dict[str, Any],
        target_atom: PolicyAtom,
    ) -> Dict[str, Any]:
        """Compute the minimal fact changes needed to satisfy the target atom's guard.

        Uses the vocabulary bridge to consider synonym and entailment
        equivalences, so that e.g. ``is_fastener`` satisfies a guard
        requiring ``product_type_fastener``.
        """
        return compute_fact_gap(current_facts, target_atom.guard)

    def _best_rate_from_active(
        self, active_atoms: List[PolicyAtom]
    ) -> Optional[float]:
        """Find the best (lowest) duty rate among active atoms."""
        rates = []
        for atom in active_atoms:
            rate = self.duty_rates.get(atom.source_id)
            if rate is not None:
                rates.append(rate)
        return min(rates) if rates else None
