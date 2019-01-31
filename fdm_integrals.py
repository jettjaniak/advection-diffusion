import numpy as np
import bandmat as bm

from typing import Tuple, Union

from scipy.linalg import solve_banded


def backward_euler_int(u_0: np.array, derivative_mat: Tuple[np.array, Tuple[int, int]],
                       tau: float, tol: float, int_u: Union[np.array, None] = None):
    """Schemat RRZ dla równania liniowego wraz z całkowaniem rozwiązania w locie.

    :param u_0: warunek początkowy
    :param derivative_mat: macierz układu liniowego w formie pasmowej
                           i krotka z liczbą pod- i naddiagonali
    :param tau: długość kroku
    :param tol: jeżeli funkcja jest < tol to kończymy rozwiązywanie i całkowanie
    :param int_u: początkowa wartość całki
    :return: dyskretyzacja rozwiązania
    """
    if int_u is None:
        int_u = np.zeros_like(u_0)
    u_old = u_0.copy()
    banded_matrix, (below, above) = derivative_mat

    # Chcemy rozwiązywać układ
    #   (Id - tau*D)x = u

    # Tworzymy macierz układu
    system_matrix = - tau * banded_matrix
    # Dodajemy identyczność do diagonali
    system_matrix[above] += 1

    i = 1
    while np.any(u_old[10:] >= tol):
        u_new = solve_banded((below, above), system_matrix, u_old)
        int_u += tau * (u_old + u_new)/2
        u_old = u_new
        i += 1
    else:
        print(f"Zakończono całkowanie w momencie {i*tau}, po {i} iteracji.")
    return int_u


def backward_euler_int_fast(u_0: np.array, derivative_mat: Tuple[np.array, Tuple[int, int]],
                            tau: float, tol: float, int_u: Union[np.array, None] = None):
    """Schemat RRZ dla równania liniowego wraz z całkowaniem rozwiązania w locie.

    Poprawiony o nieliczenie górnych wierszy, kiedy są już wystarczająco małe,
    ale wbrew oczekiwaniom nie jest szybszy niż zwykła wersja.

    :param u_0: warunek początkowy
    :param derivative_mat: macierz układu liniowego w formie pasmowej
                           i krotka z liczbą pod- i naddiagonali
    :param tau: długość kroku
    :param tol: jeżeli funkcja jest < tol to kończymy rozwiązywanie i całkowanie
    :param int_u: początkowa wartość całki
    :return: dyskretyzacja rozwiązania
    """
    n = len(u_0)
    if n > 15:
        half_n = int(n // 1.1 + 1)
        half_n_cond = half_n if n > half_n else 0
    else:
        half_n = half_n_cond = 0

    if int_u is None:
        int_u = np.zeros_like(u_0)
    u_old = u_0.copy()
    banded_matrix, (below, above) = derivative_mat

    # Chcemy rozwiązywać układ
    #   (Id - tau*D)x = u

    # Tworzymy macierz układu
    system_matrix = - tau * banded_matrix
    # Dodajemy identyczność do diagonali
    system_matrix[above] += 1

    i = 1
    # Dopóki górna połówka nie jest bliska zeru
    while np.any(u_old[half_n_cond:] >= tol):
        u_new = solve_banded((below, above), system_matrix, u_old)
        int_u += tau * (u_old + u_new)/2
        u_old = u_new
        i += 1
    else:
        print("Przełamanie, n =", n, flush=True)

    if half_n_cond == 0:
        return int_u
    else:
        # Obcinamy każdą z pod- nad- i diagonalę o half_n wyrazów od góry.
        # Nie wstawiamy zer w miejsca, które w zapisie pasmowym nie reprezentują
        # elementów macierzy, ale to nie szkodzi.
        ###banded_matrix = banded_matrix[:, :half_n]
        ###derivative_mat = (banded_matrix[:, :half_n], (below, above))
        # u_old zamiast u_new, bo while może się w ogóle nie wykonać,
        # jeżeli dużo wyrazów spadło poniżej tolerancji.
        bottom_int_u = backward_euler_int_fast(u_old[:half_n],
                                               (banded_matrix[:, :half_n], (below, above)),
                                               tau, tol, int_u[:half_n])

        return np.concatenate((bottom_int_u, int_u[half_n:]))


def trapezoids_int(u_0: np.array, derivative_mat: np.array, special: float,
                   tau: float, tol: float, int_u: Union[np.array, None] = None):
    """Schemat RRZ dla równania adwekcji-dyfuzji z nieciągłym warunkiem brzegowym.


    :param u_0: warunek początkowy
    :param derivative_mat: macierz układu liniowego
    :param special: wartość zastępująca warunek brzegowy w punkcie nieciągłości
    :param tau: długość kroku
    :param tol: jeżeli funkcja jest < tol to kończymy rozwiązywanie i całkowanie
    :param int_u: początkowa wartość całki
    :return: dyskretyzacja rozwiązania
    """
    if int_u is None:
        int_u = np.zeros_like(u_0)
    u_old = u_0.copy()
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
    first_right_vector = bm.dot_mv(rhs_matrix, u_0)
    first_right_vector[-1] += 0.5 * tau * special
    u_new = solve_banded(
        (below, above),
        system_matrix,
        first_right_vector
    )
    int_u += tau * (u_old + u_new) / 2

    i = 1
    while np.any(u_old[10:] >= tol):
        u_new = solve_banded(
            (below, above),
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

