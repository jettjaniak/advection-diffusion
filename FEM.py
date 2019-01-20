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


# N = 100
# h = 0.1
# u_0 = np.array([1] * (N))
# tau = 0.001
# m = 1000
# params = ex_params_1
def backward_euler_fem(N, h, u_0, tau, m, params):

    A, M = fem_matrices(make_b(params), make_sigma2(params),
                        make_sigma_sigma_prim(params), h, N)

    B = A - tau*M
    C = A  # Bu_k+1 = cu_k
    u = np.empty((m + 1, N))  # u(t,x)
    u[0] = u_0
    for i in range(1, m + 1):
        u[i] = spsolve(B, C @ u[i - 1])

    plt.imshow(u.T, origin="low", extent=[0, 10, 0, 10])
    u = np.concatenate((u, np.array([([0] * (m + 1))]).T), axis=1)
    return u


def trapezoids_fem(N, h, u_0, tau, m, params):
    A, M = fem_matrices(make_b(params), make_sigma2(params),
                        make_sigma_sigma_prim(params), h, N)
    B = A - (tau/2) * M
    C = A + (tau/2) * M # Bu_k+1 = cu_k
    u = np.empty((m + 1, N))  # u(t,x)
    u[0] = u_0
    for i in range(1, m + 1):
        u[i] = spsolve(B, C @ u[i - 1])

    plt.imshow(u.T, origin="low", extent=[0, 10, 0, 10])
    u = np.concatenate((u, np.array([([0] * (m + 1))]).T), axis=1)
    return u