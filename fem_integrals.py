from scipy.sparse import diags, csr_matrix
import numpy as np
from params import ex_params_1
from rates import make_b, make_sigma2,make_sigma_sigma_prim
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from FEM import fem_matrices
from typing import Callable, Tuple

def backward_euler_fem_int(N: int, h: float, u_0: np.array, tau: float, params, tol: float):
    """
    Schemat zamknięty Eulera dla problemu z wykorzystaniem metody elementu skończonego do
    przybliżania pochodnych po zmiennej x.
    :param N: liczba punktów w siatce zmiennej x -1
    :param h: krok siatki dla x
    :param u_0: wektor początkowy
    :param tau: krok siatki dla t
    :param tol: maksymalna dopuszczalna wartość funkcji po wykonaniu symulacji
    :param params: obiekt zawierający wszystkie parametry
    :return: wektor z całkami \int_0^\inf u(t,x)dt dla x z siatki
    """

    A, M = fem_matrices(make_b(params), make_sigma2(params),
                        make_sigma_sigma_prim(params), h, N)

    B = A - tau*M
    C = A  # Bu_k+1 = cu_k
    u_prev = u_0  # u(t,x)
    integral = np.empty((1, N))
    cond = True
    iterat = 1
    while cond:
        # następna obliczona wartość wektora
        u_next = spsolve(B, C @ u_prev)
        # wartość całek na odcinku długości tau i dodanie do wektora wyników
        integral += 0.5 * tau * (u_next + u_prev)
        # sprawdzenie warunku u < tol dla wszystkich x0
        cond = False
        for i in range((N//4)*3, N):
            if u_next[i] > tol:
                cond = True
                break
        u_prev = u_next
        if iterat%10000 == 0 :
            print(iterat)
        iterat += 1

    # TODO: trzeba dodać na końcu 0 dla początkowej maks liczby wiązań
    return integral
def trapezoids_fem_integrals(N: int, h: float, u_0: np.array, tau: float, tol: float, params):
    """
    Schemat Cranka-Nicholson dla problemu z wykorzystaniem metody elementu skończonego do
    przybliżania pochodnych po zmiennej x.
    :param N:
    :param h:
    :param u_0:
    :param tau:
    :param tol: maksymalna dopuszczalna wartość funkcji po wykonaniu symulacji
    :param params:
    :return: wektor z całkami dla u po t dla początkowych x0
    """
    A, M = fem_matrices(make_b(params), make_sigma2(params),
                        make_sigma_sigma_prim(params), h, N)
    B = A - (tau/2) * M
    C = A + (tau/2) * M # Bu_k+1 = cu_k
    u_prev = u_0
    integral = np.empty((1, N))
    cond = True
    iterat = 1
    while cond:
        # następna obliczona wartość wektora
        u_next = spsolve(B, C @ u_prev)
        # wartość całek na odcinku długości tau i dodanie do wektora wyników
        integral += 0.5 * tau * (u_next + u_prev)
        # sprawdzenie warunku u < tol dla wszystkich x0
        cond = False
        for i in range(N):
            if u_next[i] > tol:
                cond = True
                break
        u_prev = u_next
        if iterat % 10000 == 0:
            print(iterat)
        iterat += 1

    #TODO:trzeba dodać na końcu 0 dla początkowej maks liczby wiązań
    return integral


N = 200
h = 0.1
u_0 = np.array([1] * (N))
tau = 0.001
tol = 0.01
params = ex_params_1

integ = backward_euler_fem_int(N, h, u_0, tau, params, tol)
print(integ)