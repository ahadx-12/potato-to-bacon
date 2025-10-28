from __future__ import annotations
import re
from typing import Pattern, List

# Phrases that commonly introduce bypass/exceptions
BYPASS_TRIGGERS: List[str] = [
    r"\bunless\b",
    r"\bexcept\b(?:\s+that)?",
    r"\bprovided\s+that\b",
    r"\bso\s+long\s+as\b",
    r"\bsubject\s+to\b",
    r"\bshall\s+not\s+be\s+required\s+if\b",
    r"\bif\b[^.]*\bnot\s+required\b",
    r"\bnotwithstanding\b",  # treat carefully in logic
]

BYPASS_RE: Pattern[str] = re.compile("|".join(BYPASS_TRIGGERS), re.IGNORECASE)

# Financial/operational metrics often used in carve-outs
METRIC_WORDS = [
    "liquidity", "cash", "cash balance", "ebitda", "adjusted ebitda",
    "leverage", "net leverage", "interest coverage", "revenue",
    "market capitalization", "capex", "free cash flow", "covenant",
    "borrowing base", "availability"
]

# Threshold patterns ($, %, X-times)
THRESH_RE = re.compile(
    r"(?P<currency>\$)?(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?P<unit>bn|b|billion|mm|m|million|%)?"
    r"|(?P<xnum>\d+(?:\.\d+)?)\s*(?P<xunit>x|times)",
    re.IGNORECASE
)

# Simple obligation indicators/anti-indicators
OBLIGATION_POS_WORDS = [
    "shall", "must", "is required to", "are required to", "will", "agrees to", "undertakes to", "covenants to"
]
OBLIGATION_NEG_WORDS = [
    "shall not", "must not", "is not required to", "are not required to", "will not", "prohibited from"
]
