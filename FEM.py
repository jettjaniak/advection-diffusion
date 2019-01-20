from scipy.sparse import diags
from scipy.integrate import quad
from scipy.linalg import inv
import numpy as np
from params import ex_params_1
from rates import make_b, make_sigma2,make_sigma_sigma_prim
from solvers import backward_euler
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve


def fem_matrices(b, sigma2, sigma_sigma_prim, h, N):
    """
    Otrzymamy zadanie postaci A*(du/dt) = M*u, gdzie:
    - A - macierz z 1 nad i pod diagonalą, 4 na diagonali poza pierwszym wierszem - tam 2, całość
    pomnożona przez h/6
    - M = B - C - D
    - B[k,l] = integral(b(x)*phi_l'*phi_k)
    - C[k,l] = integral(sigma'(x)*phi_l'*phi_k)
    - D[k,l] = integral(1/2*sigma^2(x)*phi_l'*phi_k')
    wszystkie macierze trójdiagonalne
    :param b:
    :param sigma_sigma_prim = sigma*sigma'
    :return:
    """

    # siatka dla x
    x = []
    l = 0
    for i in range(N+1):
        x.append(l)
        l = l + h


    A = diags([[h/6]*(N-1), [h/3]+[2*h/3]*(N-1), [h/6]*(N-1)], [-1, 0, 1])

    # iterowanie po kwadratach
    M_over = [quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (h-y) + (sigma2(y)/2), 0, h)[0]]
    M_diag = [quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (y-h) - (sigma2(y)/2), 0, h)[0],
              quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (y) - (sigma2(y) / 2), 0, h)[0]]
    M_under = [quad(lambda y: (-b(y) + sigma_sigma_prim(y)) * (y) + (sigma2(y)/2), 0, h)[0]]

    for i in range(1, N-1):
        M_diag[i] += quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (y-x[i+1]) - (sigma2(y) / 2),
                          x[i], x[i+1])[0]
        M_over.append(quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (x[i+1]-y) + (sigma2(y)/2),
                           x[i], x[i+1])[0])
        M_under.append(quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (x[i]-y) + (sigma2(y)/2),
                            x[i], x[i+1])[0])
        M_diag.append(quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (y - x[i]) - (sigma2(y) / 2),
                          x[i], x[i+1])[0])

    M_diag[N-1] += quad(lambda y: (b(y) - sigma_sigma_prim(y)) * (y-x[N]) - (sigma2(y) / 2),
                          x[N-1], x[N])[0]
    M = diags([M_under, M_diag, M_over], [-1, 0, 1])


    return A, M


def fem_solving(A, M, backward):
    if backward:
        return A-M, A
    else:
        return A, A+M


def przyklad():
    N = 100
    h = 0.1
    u_0 = np.array([1]*(N-1)+[0])
    A,M = fem_matrices(make_b(ex_params_1), make_sigma2(ex_params_1), make_sigma_sigma_prim(ex_params_1), h, N)
    #A,M = fem_solving(A,M,True)
    a = (inv(A.toarray()) @ M.toarray()) * (1 / (h ** 2))
    a = backward_euler(u_0, a, 0.001, 1000)
    return a
    # n = len(u_0) - 1
    # u = np.empty((m+1, n+1))
    # u[0] = u_0
    # for i in range(1, m+1):
    #     u[i] = linear_solver(np.eye(n+1) - tau * derivative_matrix, u[i-1])
    # return u

    #0.001, 1000