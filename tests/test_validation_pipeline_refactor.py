from potatobacon.models import Equation, EquationDomain
from potatobacon.validation.validator import validate_equation

def test_statistical_temperature_positive_required():
    eq = Equation(
        domain=EquationDomain.STATISTICAL,
        lhs_str="S",
        rhs_str="k_B*log(W)",
        symbol_assumptions={"T": {},"W": {},"k_B": {}},
        symbol_metadata={"S":"entropy"}
    )
    # T not used -> no error; now add T to RHS and require positivity
    eq_T = eq.model_copy(update={"rhs_str": "k_B*log(W) + 1/T"})
    out = validate_equation(eq_T)
    assert any(r.check_name=="temperature_positive_required" and not r.is_valid for r in out)

    # Now set T positive=True -> error should disappear
    eq_T_ok = eq.model_copy(update={
        "rhs_str": "k_B*log(W) + 1/T",
        "symbol_assumptions": {"T": {"positive": True}, "W": {}, "k_B": {}}
    })
    out_ok = validate_equation(eq_T_ok)
    assert not any(r.check_name=="temperature_positive_required" and not r.is_valid for r in out_ok)

def test_quantum_energy_quantization_integer_dependency():
    eq = Equation(
        domain=EquationDomain.QUANTUM,
        lhs_str="E",
        rhs_str="h*nu + m*c**2",
        symbol_assumptions={"h": {}, "nu": {}, "m": {}, "c": {}},
        symbol_metadata={"E": "energy"}
    )
    out = validate_equation(eq)
    # Should warn: no integer symbols
    assert any(r.check_name=="energy_quantization_missing_integer_symbol" for r in out)

    eq_q = eq.model_copy(update={
        "rhs_str": "E0 + n*h*nu",   # depends on integer n
        "symbol_assumptions": {"h": {}, "nu": {}, "E0": {}, "n": {"integer": True}}
    })
    out2 = validate_equation(eq_q)
    assert not any(r.check_name=="energy_quantization_missing_integer_symbol" and not r.is_valid for r in out2)

def test_backward_compat_sympy_expression_split():
    eq = Equation(
        domain=EquationDomain.STATISTICAL,
        sympy_expression="S = k_B*log(W)"
    )
    assert eq.lhs_str == "S"
    assert eq.rhs_str.replace(" ", "") == "k_B*log(W)".replace(" ","")
