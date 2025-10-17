import sympy as sp
import pytest
from potatobacon.validation.relativistic import validate_relativistic, RelativisticError

def test_gamma_ok_sub_luminal():
    v, c = sp.symbols('v c', positive=True, real=True)
    gamma = 1/sp.sqrt(1 - (v/c)**2)
    validate_relativistic(gamma, units={"v":"m/s", "c":"m/s"})

def test_superluminal_trips_guard():
    v, c = sp.symbols('v c', positive=True, real=True)
    gamma = 1/sp.sqrt(1 - (v/c)**2)
    with pytest.raises(RelativisticError):
        # Units mark v as speed; validator will sample >c and demand a guard
        validate_relativistic(gamma, units={"v":"m/s", "c":"m/s"}, strict=True)
