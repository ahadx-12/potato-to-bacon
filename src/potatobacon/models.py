from __future__ import annotations
from enum import Enum
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, root_validator

Severity = Literal["INFO", "WARNING", "ERROR"]

class EquationDomain(str, Enum):
    CLASSICAL = "classical"
    RELATIVISTIC = "relativistic"
    QUANTUM = "quantum"
    STATISTICAL = "statistical"

class ValidationReport(BaseModel):
    check_name: str
    is_valid: bool
    message: str
    severity: Severity = "INFO"
    extra: Dict[str, Any] = Field(default_factory=dict)

class Equation(BaseModel):
    """
    Central IR handed to validation. We separate LHS/RHS and split
    SymPy assumptions from semantic metadata.
    """
    domain: EquationDomain
    # Canonical equation strings
    lhs_str: str                     # e.g., "S"
    rhs_str: str                     # e.g., "k_B*log(W)"
    # SymPy assumptions: {"T": {"positive": True}, "n": {"integer": True}}
    symbol_assumptions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Semantic tags for domain guards: {"P_i": "probability", "S": "entropy"}
    symbol_metadata: Dict[str, str] = Field(default_factory=dict)

    # --- Backward-compatibility shim (legacy payloads) ---
    # Legacy fields that might appear in older API callers:
    sympy_expression: Optional[str] = None  # "S = k_B*log(W)" or just "k_B*log(W)"
    symbol_assumptions_legacy: Optional[Dict[str, Any]] = None

    @root_validator(pre=True)
    def _bc_shim(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        If older callers pass a single 'sympy_expression' or 'symbol_assumptions'
        in legacy format, normalize into lhs_str/rhs_str & symbol_assumptions.
        """
        if "lhs_str" not in values or "rhs_str" not in values:
            expr = values.get("sympy_expression")
            if isinstance(expr, str) and "=" in expr:
                lhs, rhs = expr.split("=", 1)
                values["lhs_str"] = values.get("lhs_str", lhs.strip())
                values["rhs_str"] = values.get("rhs_str", rhs.strip())
            elif isinstance(expr, str):
                # Assume anonymous LHS; keep RHS
                values["lhs_str"] = values.get("lhs_str", "f")
                values["rhs_str"] = values.get("rhs_str", expr.strip())

        if "symbol_assumptions" not in values and "symbol_assumptions_legacy" in values:
            val = values.get("symbol_assumptions_legacy") or {}
            # Legacy could be {"T": "positive"} – upgrade to {"T": {"positive": True}}
            upgraded = {}
            for k, v in val.items():
                if isinstance(v, dict):
                    upgraded[k] = v
                elif isinstance(v, str):
                    upgraded[k] = {v: True}
                else:
                    upgraded[k] = {}
            values["symbol_assumptions"] = upgraded

        # Final sanity defaults:
        values.setdefault("lhs_str", "f")
        values.setdefault("rhs_str", "0")
        values.setdefault("symbol_assumptions", {})
        values.setdefault("symbol_metadata", {})
        return values
