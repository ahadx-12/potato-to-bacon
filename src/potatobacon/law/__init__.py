"""Metrics and data interfaces for the CALE-LAW tax dashboard."""

from .features_hierarchy import precedence_weight, temporal_priority, directness_score
from .contradiction_score import ContradictionFeatures, contradiction_probability
from .ambiguity_entropy import normalized_entropy
from .judicial_uncertainty import judicial_uncertainty
from .network_fragility import compute_network_scores
from .impact_weight import impact_weight
from .flaw_score import flaw_score, policy_flaw_score

__all__ = [
    "precedence_weight",
    "temporal_priority",
    "directness_score",
    "ContradictionFeatures",
    "contradiction_probability",
    "normalized_entropy",
    "judicial_uncertainty",
    "compute_network_scores",
    "impact_weight",
    "flaw_score",
    "policy_flaw_score",
]
