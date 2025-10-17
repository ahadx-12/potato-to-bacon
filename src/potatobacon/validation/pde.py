from __future__ import annotations
from enum import Enum
from typing import List, Optional
import sympy as sp
from potatobacon.semantics.operators import extract_derivative_orders


class PDEClass(str, Enum):
    ODE = "ODE"
    ELLIPTIC = "ELLIPTIC"
    PARABOLIC = "PARABOLIC"
    HYPERBOLIC = "HYPERBOLIC"
    UNKNOWN = "UNKNOWN"


def classify_pde(
    eq: sp.Basic | sp.Equality, space_vars: List[sp.Symbol], time_var: Optional[sp.Symbol]
) -> PDEClass:
    """
    MVP heuristic classification:
      - HYPERBOLIC: highest time order == 2 and some even-order spatial derivatives
      - PARABOLIC : highest time order == 1 and some second-order spatial derivatives
      - ELLIPTIC  : spatial second order only, no time derivatives
      - ODE       : no spatial derivatives at all
      - UNKNOWN   : anything else / mixed cases
    """
    expr = eq.lhs - eq.rhs if isinstance(eq, sp.Equality) else eq
    orders = extract_derivative_orders(expr, space_vars, time_var)

    max_t = orders.get(str(time_var), 0) if time_var else 0
    max_space = max((orders.get(str(s), 0) for s in space_vars), default=0)
    any_space = any(orders.get(str(s), 0) > 0 for s in space_vars)

    if not any_space and max_space == 0:
        return PDEClass.ODE

    if max_t == 0 and max_space >= 2 and max_space % 2 == 0:
        return PDEClass.ELLIPTIC

    if max_t == 1 and max_space == 2:
        return PDEClass.PARABOLIC

    if max_t == 2 and max_space >= 2 and max_space % 2 == 0:
        return PDEClass.HYPERBOLIC

    return PDEClass.UNKNOWN
