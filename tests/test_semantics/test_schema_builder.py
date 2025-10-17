"""Tests for schema building from theory IR."""

import json
from pathlib import Path

import pytest

from potatobacon.parser import DSLParser
from potatobacon.semantics import Canonicalizer
from potatobacon.semantics.schema_builder import SchemaBuilder


@pytest.fixture
def parser() -> DSLParser:
    return DSLParser()


@pytest.fixture
def canonicalizer() -> Canonicalizer:
    return Canonicalizer()


@pytest.fixture
def builder() -> SchemaBuilder:
    return SchemaBuilder()


def test_build_schema_kinetic_energy(
    parser: DSLParser, canonicalizer: Canonicalizer, builder: SchemaBuilder
) -> None:
    dsl = """
@classical @mechanics
def kinetic_energy(m: mass, v: velocity) -> energy:
    return 0.5 * m * v**2
"""
    equation = parser.parse(dsl)
    ir = canonicalizer.canonicalize(equation)
    schema = builder.build_schema(ir)
    assert schema["version"] == "1.0"
    assert schema["metadata"]["name"] == "kinetic_energy"
    assert "mechanics" in schema["metadata"]["tags"]
    assert "m" in schema["parameters"]
    assert schema["parameters"]["v"]["dimensions"]["length"] == 1
    assert schema["parameters"]["v"]["dimensions"]["time"] == -1
    assert schema["fields"]["kinetic_energy"]["dimensions"]["mass"] == 1
    assert "canonical" in schema["equation"]
    assert "sympy" in schema["equation"]


def test_schema_is_deterministic(
    parser: DSLParser, canonicalizer: Canonicalizer, builder: SchemaBuilder
) -> None:
    dsl = """
@classical
def test(a: mass, b: velocity) -> momentum:
    return a * b
"""
    schema1 = builder.build_schema(canonicalizer.canonicalize(parser.parse(dsl)))
    schema2 = builder.build_schema(canonicalizer.canonicalize(parser.parse(dsl)))
    assert json.dumps(schema1, sort_keys=True) == json.dumps(schema2, sort_keys=True)


def test_schema_includes_tags(
    parser: DSLParser, canonicalizer: Canonicalizer, builder: SchemaBuilder
) -> None:
    dsl = """
@classical @mechanics @dynamics @kinematics
def test(x: length) -> length:
    return 2.0 * x
"""
    schema = builder.build_schema(canonicalizer.canonicalize(parser.parse(dsl)))
    tags = schema["metadata"]["tags"]
    assert len(tags) == 3
    assert tags == sorted(tags)


def test_schema_dimensions_format(
    parser: DSLParser, canonicalizer: Canonicalizer, builder: SchemaBuilder
) -> None:
    dsl = """
@classical
def test(F: force) -> force:
    return 2.0 * F
"""
    schema = builder.build_schema(canonicalizer.canonicalize(parser.parse(dsl)))
    dims = schema["parameters"]["F"]["dimensions"]
    assert dims["mass"] == 1
    assert dims["length"] == 1
    assert dims["time"] == -2


def test_schema_validation(builder: SchemaBuilder) -> None:
    valid_schema = {
        "version": "1.0",
        "metadata": {"name": "test", "domain": "classical"},
        "fields": {},
        "parameters": {},
        "equation": {"canonical": "x = 1", "sympy": "1"},
    }
    assert builder.validate_schema(valid_schema)
    invalid_schema = {
        "version": "1.0",
        "metadata": {"domain": "classical"},
        "fields": {},
        "parameters": {},
        "equation": {},
    }
    assert not builder.validate_schema(invalid_schema)


def test_schema_includes_latex(
    parser: DSLParser, canonicalizer: Canonicalizer, builder: SchemaBuilder
) -> None:
    dsl = """
@classical
def test(x: mass) -> mass:
    return 2.0 * x
"""
    schema = builder.build_schema(canonicalizer.canonicalize(parser.parse(dsl)))
    assert "latex" in schema["equation"]


def test_schema_for_newton_second_law(
    parser: DSLParser, canonicalizer: Canonicalizer, builder: SchemaBuilder
) -> None:
    dsl = """
@classical @mechanics @dynamics
def newton_second_law(m: mass, a: acceleration) -> force:
    return m * a
"""
    schema = builder.build_schema(canonicalizer.canonicalize(parser.parse(dsl)))
    assert schema["metadata"]["name"] == "newton_second_law"
    assert len(schema["parameters"]) == 2
    assert schema["fields"]["newton_second_law"]["dimensions"]["mass"] == 1
    assert schema["fields"]["newton_second_law"]["dimensions"]["length"] == 1
    assert schema["fields"]["newton_second_law"]["dimensions"]["time"] == -2


def test_schema_save_and_load(
    tmp_path: Path, parser: DSLParser, canonicalizer: Canonicalizer, builder: SchemaBuilder
) -> None:
    dsl = """
@classical
def test(x: mass) -> mass:
    return 2.0 * x
"""
    schema = builder.build_schema(canonicalizer.canonicalize(parser.parse(dsl)))
    path = tmp_path / "schema.json"
    builder.save_schema(schema, path)
    loaded = builder.load_schema(path)
    assert loaded == schema
