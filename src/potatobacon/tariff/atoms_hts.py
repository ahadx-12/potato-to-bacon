from __future__ import annotations

from functools import lru_cache
from typing import List

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.hts_ingest.ingest import HTSIngestResult, load_hts_policy_atoms


@lru_cache(maxsize=1)
def _cached_ingest() -> HTSIngestResult:
    return load_hts_policy_atoms()


def tariff_policy_atoms() -> List[PolicyAtom]:
    """Return the ingested HTS policy atoms."""

    return list(_cached_ingest().atoms)


DUTY_RATES = dict(_cached_ingest().duty_rates)

__all__ = ["DUTY_RATES", "tariff_policy_atoms"]
