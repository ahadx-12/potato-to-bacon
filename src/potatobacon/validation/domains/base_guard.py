from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
import sympy as sp
from sympy.core.expr import Expr
from potatobacon.models import Equation, ValidationReport

class DomainGuard(ABC):
    """
    Guards are stateless. They receive the parsed SymPy expression (RHS)
    and the Equation (for metadata) and return a list of ValidationReport.
    """
    @abstractmethod
    def validate(self, equation: Equation, symbols: Dict[str, sp.Symbol], expr: Expr) -> List[ValidationReport]:
        raise NotImplementedError
