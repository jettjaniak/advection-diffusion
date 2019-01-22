import numpy as np
import bandmat as bm
import matplotlib.pyplot as plt

from typing import Callable, Tuple

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from scipy.integrate import quad

from rates import make_b, make_sigma2, make_sigma_sigma_prim


def fem_matrices(b: Callable, sigma2: Callable, sigma_sigma_prim: Callable, h: float, N: int) \
        -> Tuple[Tuple[np.array, Tuple[int, int]], Tuple[np.array, Tuple[int, int]]]:
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
    :param sigma2:
    :param sigma_sigma_prim = sigma*sigma'
    :param h: długość kroku dla siatki
    :param N: liczba punktów siatki -1
    :return: macierze A i M spełniające A*(du/dt) = M*u
    """

    # siatka dla x
    x = []
    l = 0
    for i in range(N+1):
        x.append(l)
        l = l + h


    #A = diags([[h/6]*(N-1), [h/3]+[2*h/3]*(N-1), [h/6]*(N-1)], [-1, 0, 1])
    A = np.array([
        [0] + [h / 6] * (N - 1),
        [h / 3] + [2 * h / 3] * (N - 1),
        [h / 6] * (N - 1) + [0]
    ])

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

    # Macierz trójdiagonalna w postaci pasmowej
    M = np.array([
        [0] + M_over,
        M_diag,
        M_under + [0]
    ])
    # Zwracamy pary (macierz, (liczba poddiagonali, liczba naddiagonali))
    return (A, (1, 1)), (M, (1, 1))



def backward_euler_fem(N: int, h: float, u_0: np.array, tau: float, m: int, params):
    """
    Schemat zamknięty Eulera dla problemu z wykorzystaniem metody elementu skończonego do
    przybliżania pochodnych po zmiennej x.
    :param N: liczba punktów w siatce zmiennej x -1
    :param h: krok siatki dla x
    :param u_0: wektor początkowy
    :param tau: krok siatki dla t
    :param m: liczba punktów w siatce dla t -1
    :param params: obiekt zawierający wszystkie parametry
    :return: macierz z przybliżonymi wartościami funkcji u w punktach siatki
    """

    (A, (A_below, A_above)), (M, (M_below, M_above)) = fem_matrices(
        make_b(params), make_sigma2(params),
        make_sigma_sigma_prim(params), h, N
    )

    # Chcemy rozwiązywać układ
    #   (A - tau*M)x = A*u

    # Tworzymy macierz układu
    # Tutaj zakładamy, że A i M mają taki sam kształt w formie pasmowej,
    # ogólnie trzeba użyć funkcji z bm.
    system_matrix = A - tau * M

    # Tworzymy instancję BandMat, żeby przemnażać przez u
    rhs_matrix = bm.band_c_bm(A_below, A_above, A)

    u = np.empty((m + 1, N))  # u(t,x)
    u[0] = u_0
    for i in range(1, m + 1):
        u[i] = solve_banded(
            # Kształt macierzy układu jest taki sam jak macierzy A
            (A_below, A_above),
            system_matrix,
            # Mnożenie macierzy pasmowej przez wektor
            bm.dot_mv(rhs_matrix, u[i - 1])
        )

    #plt.imshow(u.T, origin="low", extent=[0, 10, 0, 10])
    u = np.concatenate((u, np.array([([0] * (m + 1))]).T), axis=1)
    return u


def trapezoids_fem(N: int, h: float, u_0: np.array, tau: float, m: int, params):
    """
    Schemat Cranka-Nicholson dla problemu z wykorzystaniem metody elementu skończonego do
    przybliżania pochodnych po zmiennej x.
    :param N:
    :param h:
    :param u_0:
    :param tau:
    :param m:
    :param params:
    :return:
    """
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
