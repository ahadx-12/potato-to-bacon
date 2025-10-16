"""Utilities for building JSON schemas from theory IR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import sympy as sp

from ..core.types import Variable
from ..core.dimensions import Dimension
from .ir import TheoryIR


class SchemaBuilder:
    """Create deterministic JSON-serialisable schemas from theory IR objects."""

    def __init__(self, schema_version: str = "1.0") -> None:
        self.schema_version = schema_version

    def build_schema(self, ir: TheoryIR) -> Dict[str, Any]:
        schema = {
            "version": self.schema_version,
            "metadata": self._build_metadata(ir),
            "fields": self._build_fields(ir),
            "parameters": self._build_parameters(ir),
            "equation": self._build_equation(ir),
        }
        return schema

    # ------------------------------------------------------------------
    # Section builders

    def _build_metadata(self, ir: TheoryIR) -> Dict[str, Any]:
        equation = ir.equation
        metadata: Dict[str, Any] = {
            "name": equation.name,
            "domain": equation.domain.value,
        }

        if equation.tags:
            metadata["tags"] = sorted(equation.tags)
        if equation.description:
            metadata["description"] = equation.description
        if equation.assumptions:
            metadata["assumptions"] = equation.assumptions

        return metadata

    def _build_fields(self, ir: TheoryIR) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        for output in ir.outputs:
            fields[output.name] = self._variable_to_schema(output)
        return fields

    def _build_parameters(self, ir: TheoryIR) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {}
        for input_var in ir.inputs:
            parameters[input_var.name] = self._variable_to_schema(input_var)
        return parameters

    def _variable_to_schema(self, variable: Variable) -> Dict[str, Any]:
        schema: Dict[str, Any] = {
            "dimensions": self._dimension_to_dict(variable.dimensions),
            "unit": variable.unit or "SI",
        }
        if variable.description:
            schema["description"] = variable.description
        if variable.constraints:
            schema["constraints"] = variable.constraints
        if variable.default_value is not None:
            schema["default"] = variable.default_value
        return schema

    def _dimension_to_dict(self, dimension: Dimension) -> Dict[str, int]:
        return {
            "length": dimension.length,
            "mass": dimension.mass,
            "time": dimension.time,
            "current": dimension.current,
            "temperature": dimension.temperature,
            "amount": dimension.amount,
            "luminosity": dimension.luminosity,
        }

    def _build_equation(self, ir: TheoryIR) -> Dict[str, Any]:
        equation = {
            "canonical": ir.canonical_str,
            "sympy": str(ir.simplified_expr),
        }
        try:
            equation["latex"] = sp.latex(ir.simplified_expr)
        except Exception:  # pragma: no cover - latex failures are rare but possible
            equation["latex"] = None
        return equation

    # ------------------------------------------------------------------
    # Validation helpers

    def validate_schema(self, schema: Dict[str, Any]) -> bool:
        required_keys = ["version", "metadata", "fields", "parameters", "equation"]
        for key in required_keys:
            if key not in schema:
                return False
        metadata = schema.get("metadata", {})
        if "name" not in metadata or "domain" not in metadata:
            return False
        return True

    # ------------------------------------------------------------------
    # IO helpers

    def save_schema(self, schema: Dict[str, Any], filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(schema, indent=2, sort_keys=True))

    def load_schema(self, filepath: Path) -> Dict[str, Any]:
        return json.loads(filepath.read_text())
