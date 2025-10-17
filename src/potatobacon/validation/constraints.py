from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Iterable
import math
import numpy as np
import sympy as sp


@dataclass
class ConstraintViolation:
    kind: str
    symbol: Optional[str]
    message: str


class ConstraintError(Exception):
    def __init__(self, violations: Iterable[ConstraintViolation]):
        self.violations = list(violations)
        super().__init__("\n".join(v.message for v in self.violations))


def _sample_points(lo: float, hi: float, n: int = 5) -> np.ndarray:
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        lo, hi = -1.0, 1.0
    return np.linspace(lo, hi, n)


def _symbol_bounds(sym: sp.Symbol, constraints: Dict[str, Any]) -> Tuple[float, float]:
    # Optional explicit per-symbol range constraints (e.g., {"theta":{"range":[-sp.pi, sp.pi]}})
    c = constraints.get(str(sym), {})
    rng = c.get("range")
    if rng and len(rng) == 2:
        lo = float(sp.N(rng[0]))
        hi = float(sp.N(rng[1]))
        return lo, hi
    # Use assumptions for a sane default range
    if sym.is_positive:
        return (1e-3, 1.0)
    if sym.is_nonnegative:
        return (0.0, 1.0)
    return (-1.0, 1.0)


def validate_constraints(
    expr: sp.Basic, constraints: Dict[str, Any] | None = None, require_real: bool = True
) -> None:
    """
    Checks:
      - symbol-level: positive / nonnegative / range [lo, hi]
      - expression-level: real-valued if require_real
    Uses SymPy assumptions; when undecidable, does small numeric sampling.
    """
    constraints = constraints or {}
    violations = []

    # 1) Real-valued expression requirement (symbolic first)
    if require_real and expr.is_real is False:
        violations.append(
            ConstraintViolation(
                "realness", None, "Expression may be complex (failed symbolic realness check)."
            )
        )

    # 2) Symbol-level simple checks via assumptions
    for sym in sorted(expr.free_symbols, key=lambda s: s.name):
        symc = constraints.get(str(sym), {})
        if symc.get("positive") and sym.is_positive is False:
            violations.append(
                ConstraintViolation(
                    "positive", str(sym), f"{sym} must be positive (assumption violated)."
                )
            )
        if symc.get("nonnegative") and sym.is_nonnegative is False:
            violations.append(
                ConstraintViolation(
                    "nonnegative", str(sym), f"{sym} must be nonnegative (assumption violated)."
                )
            )

    # 3) Numeric sampling fallback for undecidable constraints
    # Build a lambdified function of all free symbols
    free_syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
    if free_syms:
        f = sp.lambdify(free_syms, expr, modules=["numpy"])
        # For each symbol, pick bounds from constraints or assumptions
        grids = []
        for s in free_syms:
            lo, hi = _symbol_bounds(s, constraints)
            grids.append(_sample_points(lo, hi))

        # Construct a minimal grid of sample points (diagonal sweep)
        for i in range(min(5, max(len(g) for g in grids))):
            vals = []
            for g in grids:
                vals.append(g[i % len(g)])
            try:
                val = f(*vals)
            except Exception:
                continue
            if require_real and (np.iscomplexobj(val)):
                violations.append(
                    ConstraintViolation(
                        "realness", None, "Expression returned complex values on sample points."
                    )
                )
                break

            # Per-symbol numeric positivity/nonnegativity checks
            for s in free_syms:
                sc = constraints.get(str(s), {})
                v = vals[free_syms.index(s)]
                if sc.get("positive") and not (v > 0):
                    violations.append(
                        ConstraintViolation(
                            "positive", str(s), f"{s} sampled value {v} is not positive."
                        )
                    )
                    break
                if sc.get("nonnegative") and not (v >= 0):
                    violations.append(
                        ConstraintViolation(
                            "nonnegative", str(s), f"{s} sampled value {v} is negative."
                        )
                    )
                    break

    if violations:
        raise ConstraintError(violations)
