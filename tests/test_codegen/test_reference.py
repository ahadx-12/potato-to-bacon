import sympy as sp
from potatobacon.codegen.reference import generate_numpy


def test_codegen_ke_runs():
    m, v = sp.symbols("m v", positive=True, real=True)
    E = sp.Rational(1, 2) * m * v**2
    code = generate_numpy(E, name="ke")
    ns = {}
    exec(code, ns, ns)
    assert abs(ns["ke"](2.0, 3.0) - 0.5 * 2.0 * 9.0) < 1e-9  # 9
