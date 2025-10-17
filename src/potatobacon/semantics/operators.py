from __future__ import annotations
from typing import Dict, Tuple, Iterable
import sympy as sp

def extract_derivative_orders(expr: sp.Basic,
                              space_vars: Iterable[sp.Symbol],
                              time_var: sp.Symbol | None) -> Dict[str, int]:
    """
    Returns highest derivative order seen per var: {"t": 2, "x": 2, ...}
    """
    orders: Dict[str, int] = {}
    space_set = set(space_vars)
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative):
            # node.variables is a tuple like (x, x, t) for d^3/dx^2 dt
            counts: Dict[sp.Symbol, int] = {}
            for v in node.variables:
                counts[v] = counts.get(v, 0) + 1
            for v, k in counts.items():
                name = str(v)
                orders[name] = max(orders.get(name, 0), k)
    # ensure keys exist even if 0
    if time_var is not None:
        orders.setdefault(str(time_var), 0)
    for s in space_set:
        orders.setdefault(str(s), 0)
    return orders
