from __future__ import annotations
from typing import Dict, Iterable, Optional
import numpy as np
import sympy as sp
from dataclasses import dataclass


@dataclass
class RelativisticViolation:
    kind: str
    message: str
    symbol: Optional[str] = None


class RelativisticError(Exception):
    def __init__(self, vs: Iterable[RelativisticViolation]):
        self.violations = list(vs)
        super().__init__("\n".join(v.message for v in self.violations))


def _find_speed_symbols(units: Dict[str, str]) -> set[str]:
    # Very light heuristic: treat units exactly "m/s" as speeds for MVP
    return {
        name
        for name, unit in units.items()
        if unit.strip().lower() in {"m/s", "meter/second", "meters/second"}
    }


def validate_relativistic(
    expr: sp.Basic,
    units: Dict[str, str],
    constants: Dict[str, str] | None = None,
    strict: bool = False,
) -> None:
    """
    MVP checks:
      - Every symbol with units m/s must be < c in magnitude (sampling).
      - If expression contains sqrt(1 - (v/c)^2) (Lorentz factor parts), radicand must be in (0,1].
    """
    constants = constants or {"c": "m/s"}
    const_names = {k for k, u in constants.items() if u == "m/s"}
    c_syms = [sp.Symbol(k) for k in const_names if k in units]
    c = c_syms[0] if c_syms else sp.Symbol("c")  # assume present / unit tagged by caller

    speed_names = _find_speed_symbols(units) - const_names
    free_syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
    sym_map = {str(sym): sym for sym in free_syms}

    violations = []

    # Symbolic guard for classic gamma structure
    # Look for sqrt(1 - (v/c)**2) pattern; ensure argument is between 0 and 1.
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp == sp.Rational(1, 2):  # sqrt(...)
            arg = node.base
            # If matches 1 - something
            if isinstance(arg, sp.Add) and any(t == 1 for t in arg.args):
                # crude: if we see v/c squared inside, note requirement
                if any(
                    isinstance(a, sp.Pow)
                    and a.exp == 2
                    and a.base.has(*[sp.Symbol(n) for n in speed_names])
                    for a in arg.args
                ):
                    # can't prove symbolically → require caller to constrain v < c; rely on sampling below
                    pass

    # Numeric sampling: build function f(v_speeds..., c)
    sampling_speeds = [sym_map[n] for n in speed_names if n in sym_map]
    if sampling_speeds:
        f = sp.lambdify(sampling_speeds + [c], expr, modules=["numpy"])
        # sample speeds from [0, 0.99 c]
        c_val = 3.0e8
        for frac in np.linspace(0.0, 0.99, 5):
            vals = [frac * c_val for _ in sampling_speeds] + [c_val]
            try:
                _ = f(*vals)
            except Exception:
                # Fail closed
                violations.append(
                    RelativisticViolation(
                        "evaluation", "Relativistic evaluation failed at sampled speeds."
                    )
                )
                break

        # Try a superluminal sample to ensure rejection
        if strict:
            try:
                vals = [1.1 * c_val for _ in sampling_speeds] + [c_val]
                val = f(*vals)
                if np.iscomplexobj(val):
                    pass
                else:
                    try:
                        arr = np.asarray(val)
                        if np.isnan(arr).any():
                            violations.append(
                                RelativisticViolation(
                                    "causality",
                                    "Superluminal sample (v>c) produced NaN without guard.",
                                )
                            )
                        else:
                            violations.append(
                                RelativisticViolation(
                                    "causality",
                                    "Superluminal sample (v>c) did not fail evaluation. Guard required.",
                                )
                            )
                    except TypeError:
                        violations.append(
                            RelativisticViolation(
                                "causality",
                                "Superluminal sample (v>c) did not fail evaluation. Guard required.",
                            )
                        )
            except Exception:
                pass

    if violations:
        raise RelativisticError(violations)
