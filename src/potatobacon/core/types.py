"""Core type definitions used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import sympy as sp

from .dimensions import Dimension


class EquationDomain(str, Enum):
    CLASSICAL = "classical"
    RELATIVISTIC = "relativistic"
    QUANTUM = "quantum"
    STATISTICAL = "statistical"


class VariableRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    CONSTANT = "constant"
    INTERMEDIATE = "intermediate"


@dataclass
class Variable:
    name: str
    role: VariableRole
    dimensions: Dimension
    unit: Optional[str] = None
    description: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    default_value: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Variable name cannot be empty")
        if not isinstance(self.role, VariableRole):
            self.role = VariableRole(self.role)


@dataclass
class Equation:
    name: str
    inputs: List[Variable]
    outputs: List[Variable]
    expression: sp.Expr
    domain: EquationDomain = EquationDomain.CLASSICAL
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Equation name cannot be empty")
        if not self.inputs:
            raise ValueError("Equation must have at least one input")
        if not self.outputs:
            raise ValueError("Equation must have at least one output")
        if not isinstance(self.expression, sp.Expr):
            try:
                self.expression = sp.sympify(self.expression)
            except Exception as exc:  # pragma: no cover - defensive
                raise TypeError("Expression must be a SymPy expression") from exc
        if not isinstance(self.domain, EquationDomain):
            self.domain = EquationDomain(self.domain)

    def get_variable(self, name: str) -> Optional[Variable]:
        for var in self.inputs + self.outputs:
            if var.name == name:
                return var
        return None

    def all_variables(self) -> List[Variable]:
        return self.inputs + self.outputs


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_check_passed(self, check_name: str) -> None:
        self.checks_passed.append(check_name)


@dataclass
class TranslationResult:
    equation: Equation
    canonical_form: str
    schema: Dict[str, Any]
    validation: ValidationResult
    generated_code: Optional[str] = None
    manifest_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return self.validation.valid

    def summary(self) -> str:
        status = "✓ VALID" if self.is_valid() else "✗ INVALID"
        return f"{status} | {self.equation.name} | {self.equation.domain.value}"
