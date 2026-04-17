import sympy as sp

x = sp.symbols('x')

def Newton(f, semilla, tol):
    df = sp.diff(f, x)
    newton = x - f/df
    error = 1
    contador = 0
    while error > tol and contador <= 50:
        contador += 1
        x1 = newton.subs(x, semilla)
        error = abs(x1 - semilla)
        semilla = x1
    return float(x1)