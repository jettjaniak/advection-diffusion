from typing import Callable, Tuple
from scipy.sparse import diags, csr_matrix


def fdm(b: Callable, sigma2: Callable, h: float, n: int) -> Tuple[csr_matrix, float]:
    diagonal = [-3 / (2 * h)]
    under = []
    over1 = [2 / h]
    over2 = [-1 / (2 * h)] + [0] * (n - 3)
    for k in range(1, n):
        s_k = sigma2(h * k) / (h ** 2)
        b_k = b(h * k) / (2 * h)
        diagonal.append(-s_k)
        under.append((s_k / 2) - b_k)
        over1.append((s_k / 2) + b_k)

    return diags((under, diagonal, over1[:-1], over2), (-1, 0, 1, 2)), over1[-1]
