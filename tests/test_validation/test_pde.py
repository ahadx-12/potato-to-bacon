import sympy as sp
from potatobacon.validation.pde import classify_pde, PDEClass

def test_wave_eq_hyperbolic():
    x, t, c = sp.symbols('x t c', real=True)
    u = sp.Function('u')(x, t)
    eq = sp.Eq(sp.diff(u, t, 2), c**2 * sp.diff(u, x, 2))
    cls = classify_pde(eq, [x], t)
    assert cls == PDEClass.HYPERBOLIC

def test_heat_eq_parabolic():
    x, t, k = sp.symbols('x t k', real=True)
    u = sp.Function('u')(x, t)
    eq = sp.Eq(sp.diff(u, t), k * sp.diff(u, x, 2))
    cls = classify_pde(eq, [x], t)
    assert cls == PDEClass.PARABOLIC

def test_poisson_elliptic():
    x, y = sp.symbols('x y', real=True)
    u = sp.Function('u')(x, y)
    f = sp.Function('f')(x, y)
    eq = sp.Eq(sp.diff(u, x, 2) + sp.diff(u, y, 2), f)
    cls = classify_pde(eq, [x, y], None)
    assert cls == PDEClass.ELLIPTIC
