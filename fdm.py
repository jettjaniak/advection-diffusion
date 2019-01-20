from typing import Callable, Tuple
from scipy.sparse import diags, csr_matrix


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
