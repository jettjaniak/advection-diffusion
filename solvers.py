import numpy as np

from typing import Callable
from scipy.linalg import solve as linear_solver


def backward_euler(u_0: np.array, derivative_matrix: np.array, tau: float, m: int):
    n = len(u_0)
    u = np.empty((m+1, n))
    u[0] = u_0
    for i in range(1, m+1):
        u[i] = linear_solver(np.eye(n) - tau * derivative_matrix, u[i-1])
    return u


def trapezoids(u_0: np.array, derivative_matrix: np.array, special: float,
               tau: float, m: int):
    n = len(u_0)
    u = np.empty((m+1, n))
    u[0] = u_0

    first_right_vector = (np.eye(n) + 0.5 * tau * derivative_matrix) @ u[0]
    first_right_vector[-1] += 0.5 * tau * special
    u[1] = linear_solver(
        np.eye(n) - 0.5 * tau * derivative_matrix,
        first_right_vector
    )

    for i in range(2, m+1):
        u[i] = linear_solver(
            np.eye(n) - 0.5 * tau * derivative_matrix,
            (np.eye(n) + 0.5 * tau * derivative_matrix) @ u[i-1]
        )
    return u
