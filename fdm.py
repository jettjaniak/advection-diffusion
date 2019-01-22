import numpy as np

from typing import Callable, Tuple

from scipy import sparse
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def fdm(b: Callable, sigma2: Callable, h: float, n: int) -> Tuple[csr_matrix, float]:
    """Funkcja zwraca macierz (która pomnożona przez u daje u_t) i wartość specjalną.

    Funkcja zwraca macierz A wymiaru n x n spełniającą
    w przybliżeniu (metoda różnic skończonych)

      Au = u_t = b(x)u_x + sigma^2(x)/2 u_xx,

    oraz wartość specjalną, niezbędną w przypadku użycia schematu trapezów.

    :param b: funkcja związana z adwekcją
    :param sigma2: funkcja związana z dyfuzją
    :param h: długość kroku
    :param n: liczba kroków
    :return:
    """
    diagonal = [-3 / (2 * h)]
    under = []
    over1 = [2 / h]
    over2 = [-1 / (2 * h)] + [0] * (n - 3)
    for k in range(1, n):
        s_k = sigma2(h * k) / (h ** 2)
        b_k = b(h * k) / (2 * h)
        diagonal.append(-s_k)
        under.append((s_k / 2) - b_k)
        # W ostatnim kroku do naddiagonali dodajemy dodatkowy element.
        # Obetniemy go i zwrócimy jako wartość specjalną.
        over1.append((s_k / 2) + b_k)

    return diags((under, diagonal, over1[:-1], over2), (-1, 0, 1, 2)), over1[-1]


def backward_euler(u_0: np.array, derivative_matrix: sparse.csr_matrix, tau: float, m: int):
    """Schemat RRZ dla równania liniowego.

    :param u_0: warunek początkowy
    :param derivative_matrix: macierz układu liniowego
    :param tau: długość kroku
    :param m: liczba kroków
    :return: dyskretyzacja rozwiązania
    """
    n = len(u_0)
    u = np.empty((m+1, n))
    u[0] = u_0
    for i in range(1, m+1):
        u[i] = spsolve(sparse.eye(n) - tau * derivative_matrix, u[i-1])
    return u


def trapezoids(u_0: np.array, derivative_matrix: np.array, special: float,
               tau: float, m: int):
    """Schemat RRZ dla równania adwekcji-dyfuzji z nieciągłym warunkiem brzegowym.


    :param u_0: warunek początkowy
    :param derivative_matrix: macierz układu liniowego
    :param special: wartość zastępująca warunek brzegowy w punkcie nieciągłości
    :param tau: długość kroku
    :param m: liczba kroków
    :return: dyskretyzacja rozwiązania
    """
    n = len(u_0)
    u = np.empty((m+1, n))
    u[0] = u_0

    first_right_vector = (sparse.eye(n) + 0.5 * tau * derivative_matrix) @ u[0]
    first_right_vector[-1] += 0.5 * tau * special
    u[1] = spsolve(
        sparse.eye(n) - 0.5 * tau * derivative_matrix,
        first_right_vector
    )

    for i in range(2, m+1):
        u[i] = spsolve(
            sparse.eye(n) - 0.5 * tau * derivative_matrix,
            (sparse.eye(n) + 0.5 * tau * derivative_matrix) @ u[i-1]
        )
    return u
