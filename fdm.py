from typing import Callable
from scipy.sparse import diags, csr_matrix


def fdm(b: Callable, sigma2: Callable, h: float, n: int) -> csr_matrix:
    diagonal = [-3 / (2 * h)]
    under = []
    over1 = [2 / h]
    over2 = [-1 / (2 * h)] + [0] * (n - 2)
    for k in range(1, n):
        s_k = sigma2(h * k) / (h ** 2)
        b_k = b(h * k) / (2 * h)
        diagonal.append(-s_k)
        under.append((s_k / 2) - b_k)
        over1.append((s_k / 2) + b_k)
    under.append(0)
    diagonal.append(1)
    return diags((under, diagonal, over1, over2), (-1, 0, 1, 2))
