"""A very small subset of SymPy used for unit tests.

This module implements enough symbolic behaviour to satisfy the project's
unit tests without pulling in the full SymPy dependency.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Tuple, Union

NumberLike = Union[int, float, "Number"]


def _to_expr(value: Union["Expr", NumberLike]) -> "Expr":
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float)):
        return Number(value)
    raise TypeError(f"Cannot convert {type(value)} to Expr")


class Expr:
    """Base class for all symbolic expressions."""

    args: Tuple["Expr", ...] = ()

    def __add__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Add(self, _to_expr(other))

    def __radd__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Add(_to_expr(other), self)

    def __sub__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Add(self, Mul(Number(-1), _to_expr(other)))

    def __rsub__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Add(_to_expr(other), Mul(Number(-1), self))

    def __mul__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Mul(self, _to_expr(other))

    def __rmul__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Mul(_to_expr(other), self)

    def __truediv__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Div(self, _to_expr(other))

    def __rtruediv__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Div(_to_expr(other), self)

    def __pow__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Pow(self, _to_expr(other))

    def __rpow__(self, other: Union["Expr", NumberLike]) -> "Expr":
        return Pow(_to_expr(other), self)

    def __neg__(self) -> "Expr":
        return Mul(Number(-1), self)

    def __pos__(self) -> "Expr":
        return self

    def has(self, cls: type) -> bool:
        if isinstance(self, cls):
            return True
        return any(isinstance(arg, Expr) and arg.has(cls) for arg in self.args)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - structural comparison
        if type(self) is not type(other):
            return False
        return getattr(self, "_eq_args", self.args) == getattr(other, "_eq_args", other.args)

    def __str__(self) -> str:
        raise NotImplementedError

    def _str(self, parent_prec: int = 0) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Number(Expr):
    value: float

    def __init__(self, value: NumberLike):
        object.__setattr__(self, "value", float(value))

    @property
    def args(self) -> Tuple[()]:  # type: ignore[override]
        return ()

    def __str__(self) -> str:
        if self.value.is_integer():
            return str(int(self.value))
        return repr(self.value)

    def _str(self, parent_prec: int = 0) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Number):
            return abs(self.value - other.value) < 1e-12
        if isinstance(other, (int, float)):
            return abs(self.value - float(other)) < 1e-12
        return False


@dataclass(frozen=True)
class Symbol(Expr):
    name: str

    @property
    def args(self) -> Tuple[()]:  # type: ignore[override]
        return ()

    def __str__(self) -> str:
        return self.name

    def _str(self, parent_prec: int = 0) -> str:
        return self.name


class BinaryOp(Expr):
    precedence: int = 0
    operator: str = ""

    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right
        self.args = (left, right)

    def __str__(self) -> str:
        return self._str()

    def _str(self, parent_prec: int = 0) -> str:
        left_str = self.left._str(self.precedence)
        right_str = self.right._str(self.precedence - (1 if self.operator == "**" else 0))
        text = f"{left_str}{self.operator}{right_str}"
        if self.precedence < parent_prec:
            return f"({text})"
        return text


class Add(BinaryOp):
    precedence = 10
    operator = "+"


class Mul(BinaryOp):
    precedence = 20
    operator = "*"


class Div(BinaryOp):
    precedence = 20
    operator = "/"


class Pow(BinaryOp):
    precedence = 30
    operator = "**"


def Float(value: Union[str, float, int]) -> Number:
    return Number(float(value))


def sympify(value: Union[str, Expr, NumberLike]) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float)):
        return Number(value)
    if isinstance(value, str):
        node = ast.parse(value, mode="eval").body
        return simplify(_from_ast(node))
    raise TypeError(f"Unsupported value for sympify: {type(value)}")


def _from_ast(node: ast.AST) -> Expr:
    if isinstance(node, ast.BinOp):
        left = _from_ast(node.left)
        right = _from_ast(node.right)
        if isinstance(node.op, ast.Add):
            return Add(left, right)
        if isinstance(node.op, ast.Sub):
            return Add(left, Mul(Number(-1), right))
        if isinstance(node.op, ast.Mult):
            return Mul(left, right)
        if isinstance(node.op, ast.Div):
            return Div(left, right)
        if isinstance(node.op, ast.Pow):
            return Pow(left, right)
        raise ValueError(f"Unsupported binary operator: {node.op}")
    if isinstance(node, ast.UnaryOp):
        operand = _from_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary operator: {node.op}")
    if isinstance(node, ast.Constant):
        return Number(node.value)
    if isinstance(node, ast.Name):
        return Symbol(node.id)
    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def simplify(expr: Expr) -> Expr:
    if isinstance(expr, Number) or isinstance(expr, Symbol):
        return expr
    if isinstance(expr, Add):
        left = simplify(expr.left)
        right = simplify(expr.right)
        if isinstance(right, Mul) and isinstance(right.left, Number) and abs(right.left.value + 1) < 1e-12:
            if right.right == left:
                return Number(0)
        if isinstance(left, Mul) and isinstance(left.left, Number) and abs(left.left.value + 1) < 1e-12:
            if left.right == right:
                return Number(0)
        if isinstance(left, Number) and isinstance(right, Number):
            return Number(left.value + right.value)
        return Add(left, right)
    if isinstance(expr, Mul):
        left = simplify(expr.left)
        right = simplify(expr.right)
        if isinstance(left, Number) and isinstance(right, Number):
            return Number(left.value * right.value)
        return Mul(left, right)
    if isinstance(expr, Div):
        left = simplify(expr.left)
        right = simplify(expr.right)
        if isinstance(left, Number) and isinstance(right, Number):
            return Number(left.value / right.value)
        return Div(left, right)
    if isinstance(expr, Pow):
        left = simplify(expr.left)
        right = simplify(expr.right)
        if isinstance(left, Number) and isinstance(right, Number):
            return Number(left.value ** right.value)
        return Pow(left, right)
    return expr


def latex(expr: Expr) -> str:  # pragma: no cover - simple passthrough
    return str(expr)
