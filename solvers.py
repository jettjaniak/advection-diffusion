import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve


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
