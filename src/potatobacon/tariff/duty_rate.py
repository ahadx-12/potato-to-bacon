from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DutyRate:
    type: Literal["ad_valorem", "free", "compound", "specific", "unknown"]
    ad_valorem: float | None = None
    specific: float | None = None
    specific_unit: str | None = None
    raw: str = ""
