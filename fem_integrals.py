import numpy as np
import bandmat as bm

from typing import Union

from scipy.linalg import solve_banded

from rates import make_b, make_sigma2, make_sigma_sigma_prim
from FEM import fem_matrices


def backward_euler_fem_int(N: int, h: float, u_0: np.array, params, tau: float,
                           tol: float, int_u: Union[np.array, None] = None):
    """
    Schemat zamknięty Eulera dla problemu z wykorzystaniem metody elementu skończonego do
    przybliżania pochodnych po zmiennej x.
    :param N: liczba punktów w siatce zmiennej x -1
    :param h: krok siatki dla x
    :param u_0: wektor początkowy
    :param tau: krok siatki dla t
    :param params: obiekt zawierający wszystkie parametry
    :param tol: jeżeli funkcja jest < tol to kończymy rozwiązywanie i całkowanie
    :param int_u: początkowa wartość całki
    :return: macierz z przybliżonymi wartościami funkcji u w punktach siatki
    """
    if int_u is None:
        int_u = np.zeros_like(u_0)
    u_old = u_0.copy()
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

    i = 1
    while np.any(u_old >= tol):
        u_new = solve_banded(
            # Kształt macierzy układu jest taki sam jak macierzy A
            (A_below, A_above),
            system_matrix,
            # Mnożenie macierzy pasmowej przez wektor
            bm.dot_mv(rhs_matrix, u_old)
        )
        int_u += tau * (u_old + u_new) / 2
        u_old = u_new
        i += 1

    return int_u


def trapezoids_fem_int(N: int, h: float, u_0: np.array, params, tau: float,
                       tol: float, int_u: Union[np.array, None] = None):
    """
    Schemat Cranka-Nicholson dla problemu z wykorzystaniem metody elementu skończonego do
    przybliżania pochodnych po zmiennej x.
    :param N:
    :param h:
    :param u_0:
    :param params:
    :param tau:
    :param tol: jeżeli funkcja jest < tol to kończymy rozwiązywanie i całkowanie
    :param int_u: początkowa wartość całki
    :return:
    """
    if int_u is None:
        int_u = np.zeros_like(u_0)
    u_old = u_0.copy()
    (A, (A_below, A_above)), (M, (M_below, M_above)) = fem_matrices(
        make_b(params), make_sigma2(params),
        make_sigma_sigma_prim(params), h, N
    )

    # Chcemy rozwiązywać układ
    #   (A - 0.5*tau*M)x = (A + 0.5*tau*M)*u

    # Tworzymy macierz układu
    # Tutaj zakładamy, że A i M mają taki sam kształt w formie pasmowej,
    # ogólnie trzeba użyć funkcji z bm.
    system_matrix = A - 0.5 * tau * M

    # Tworzymy instancję BandMat, żeby przemnażać przez u
    rhs_matrix = A + 0.5 * tau * M
    rhs_matrix = bm.band_c_bm(A_below, A_above, rhs_matrix)

    i = 1
    while np.any(u_old >= tol):
        u_new = solve_banded(
            # Kształt macierzy układu jest taki sam jak macierzy A
            (A_below, A_above),
            system_matrix,
            # Mnożenie macierzy pasmowej przez wektor
            bm.dot_mv(rhs_matrix, u_old)
        )
        int_u += tau * (u_old + u_new)/2
        u_old = u_new
        i += 1
    else:
        print(f"Zakończono całkowanie w momencie {i*tau}, po {i} iteracji.")
    return int_u
