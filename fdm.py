import numpy as np
import bandmat as bm

from typing import Callable, Tuple

from scipy.linalg import solve_banded


def derivative_matrix(b: Callable, sigma2: Callable, h: float, n: int) ->\
        Tuple[Tuple[np.array, Tuple[int, int]], float]:
    """Funkcja zwraca macierz (która pomnożona przez u daje u_t) i wartość specjalną.

    Funkcja zwraca macierz A w formie pasmowej wymiaru n x n, która spełnia
    w przybliżeniu (metoda różnic skończonych)

      Au = u_t = b(x)u_x + sigma^2(x)/2 u_xx,

    i liczbę pod- i naddiagonali oraz wartość specjalną,
    niezbędną w przypadku użycia schematu trapezów.

    :param b: funkcja związana z adwekcją
    :param sigma2: funkcja związana z dyfuzją
    :param h: długość kroku
    :param n: liczba kroków
    :return:
    """
    diagonal = [0]
    under = []
    over1 = [0]
    for k in range(1, n):
        s_k = sigma2(h * k) / (h ** 2)
        b_k = b(h * k) / (2 * h)
        diagonal.append(-s_k)
        under.append((s_k / 2) - b_k)
        # W ostatnim kroku do naddiagonali dodajemy dodatkowy element.
        # Obetniemy go i zwrócimy jako wartość specjalną.
        over1.append((s_k / 2) + b_k)

    banded_matrix = np.array([
        [0] + over1[:-1],
        diagonal,
        under + [0]
    ])
    below = 1
    above = 1
    return (banded_matrix, (below, above)), over1[-1]


def backward_euler(u_0: np.array, derivative_mat: Tuple[np.array, Tuple[int, int]],
                   tau: float, m: int):
    """Schemat RRZ dla równania liniowego.

    :param u_0: warunek początkowy
    :param derivative_mat: macierz układu liniowego w formie pasmowej
                           i krotka z liczbą pod- i naddiagonali
    :param tau: długość kroku
    :param m: liczba kroków
    :return: dyskretyzacja rozwiązania
    """
    n = len(u_0)
    u = np.empty((m+1, n))
    u[0] = u_0
    banded_matrix, (below, above) = derivative_mat
    # Chcemy rozwiązywać układ
    #   (Id - tau*D)x = u
    # Tworzymy macierz układu
    system_matrix = - tau * banded_matrix
    # Dodajemy identyczność do diagonali
    system_matrix[above] += 1
    for i in range(1, m+1):
        u[i] = solve_banded((below, above), system_matrix, u[i-1])
    return u


def trapezoids(u_0: np.array, derivative_mat: np.array, special: float,
               tau: float, m: int):
    """Schemat RRZ dla równania adwekcji-dyfuzji z nieciągłym warunkiem brzegowym.


    :param u_0: warunek początkowy
    :param derivative_mat: macierz układu liniowego
    :param special: wartość zastępująca warunek brzegowy w punkcie nieciągłości
    :param tau: długość kroku
    :param m: liczba kroków
    :return: dyskretyzacja rozwiązania
    """
    n = len(u_0)
    u = np.empty((m+1, n))
    u[0] = u_0

    banded_matrix, (below, above) = derivative_mat
    # Chcemy rozwiązywać układ
    #   (Id - 0.5*tau*D)x = (Id + 0.5*tau*D)u.

    # Tworzymy macierz układu
    system_matrix = - 0.5 * tau * banded_matrix
    # Dodajemy identyczność do diagonali
    system_matrix[above] += 1

    # Tworzymy instancję BandMat, żeby przemnażać przez u
    rhs_matrix = 0.5 * tau * banded_matrix
    rhs_matrix[above] += 1
    rhs_matrix = bm.band_c_bm(below, above, rhs_matrix)

    # Mnożenie macierzy pasmowej przez wektor
    first_right_vector = bm.dot_mv(rhs_matrix, u[0])
    first_right_vector[-1] += 0.5 * tau * special
    u[1] = solve_banded(
        (below, above),
        system_matrix,
        first_right_vector
    )

    for i in range(2, m+1):
        u[i] = solve_banded(
            (below, above),
            system_matrix,
            # Mnożenie macierzy pasmowej przez wektor
            bm.dot_mv(rhs_matrix, u[i-1])
        )
    return u

