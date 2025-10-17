import sympy as sp
import pytest
from potatobacon.validation.constraints import validate_constraints, ConstraintError

def test_positive_mass_real_energy():
    m, v = sp.symbols('m v', positive=True, real=True)
    E = sp.Rational(1,2)*m*v**2
    validate_constraints(E, {"m":{"positive": True}, "v":{"nonnegative": True}})

def test_negative_mass_rejected():
    m, v = sp.symbols('m v', real=True)
    E = sp.Rational(1,2)*m*v**2
    with pytest.raises(ConstraintError):
        validate_constraints(E, {"m":{"positive": True}})
