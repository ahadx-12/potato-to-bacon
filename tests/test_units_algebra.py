import math

import pytest

from potatobacon.units.algebra import Quantity, UnitParseError, format_quantity, parse_unit_expr


def test_parse_unit_expr_handles_prefix_and_derived():
    newton = parse_unit_expr("N")
    combo = parse_unit_expr("kg*m/s^2")
    assert newton.dims == combo.dims
    assert math.isclose(newton.scale, combo.scale)

    kilojoule = parse_unit_expr("kJ")
    assert math.isclose(kilojoule.scale, 1000.0)
    assert kilojoule.dims == parse_unit_expr("J").dims


def test_parse_unit_expr_supports_implicit_multiplication():
    quantity = parse_unit_expr("kg m s^-2")
    assert quantity.dims == parse_unit_expr("N").dims


def test_format_quantity_prefers_named_units():
    joule = parse_unit_expr("kg*m^2/s^2")
    assert format_quantity(joule) == "J"
    scaled = Quantity(0.01, joule.dims)
    assert format_quantity(scaled).startswith("0.01")


def test_unknown_symbol_raises():
    with pytest.raises(UnitParseError):
        parse_unit_expr("unknown_unit")
