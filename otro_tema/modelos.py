import numpy as np
import sel as sl
import sympy as sp

x = sp.symbols('x')

def Matrix(x_data):
    n = len(x_data)
    A = np.zeros([n,n])
    A[0:n,0] = 1.0
    for j in range(1, n):
        for i in range(0, n):
            A[i,j] = A[i, j-1]*x_data[i]
    return A

def polinomial_simple(x_data, y_data):
    A = Matrix(x_data)
    coeficientes = sl.eliminacion_DD(A, y_data)
    Polinomio = sum(coeficientes[i] * (x**i) for i in range(len(x_data)))
    return Polinomio

def lagrange(x_data, y_data):
    sumPolinomio = 0
    for i in range(len(x_data)):
        Li = 1
        for j in range(len(x_data)):
            if j != i:
                Li *= (x-x_data[j]) / (x_data[i]-x_data[j])
        sumPolinomio += Li*y_data[i]
    return sp.expand(sumPolinomio)