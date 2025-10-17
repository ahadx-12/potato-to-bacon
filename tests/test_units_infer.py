import pytest

from potatobacon.units.infer import evaluate_equation_dimensions, infer_from_equation


def test_infer_from_equation_solves_energy():
    result, trace = infer_from_equation("E = 0.5*m*v^2", {"m": "kg", "v": "m/s"})
    assert result["m"] == "kg"
    assert result["v"] == "m/s"
    assert result["E"] == "J"
    assert any(step["action"] == "infer" and step["detail"].startswith("Solved dimensions") for step in trace)


def test_infer_from_equation_reports_underdetermined():
    result, trace = infer_from_equation("F = m*a", {"m": "kg"})
    assert result["m"] == "kg"
    assert any(step["action"] == "underdetermined" for step in trace)


def test_evaluate_equation_dimensions_requires_all_units():
    with pytest.raises(Exception):
        evaluate_equation_dimensions("F = m*a", {"m": "kg"})


def test_evaluate_equation_dimensions_returns_summary():
    units, trace, summary = evaluate_equation_dimensions("F = m*a", {"F": "N", "m": "kg", "a": "m/s^2"})
    assert units["F"] == "N"
    assert summary[0]["lhs"] == "N"
    assert trace  # known steps present
