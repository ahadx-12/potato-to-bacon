from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExtractedEvidence:
    facts: Dict[str, Any]
    provenance: Dict[str, str]
    warnings: List[str]
    confidence: float
    extraction_metadata: Dict[str, Any]
