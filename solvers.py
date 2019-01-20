import numpy as np

from scipy.linalg import solve as linear_solver


def backward_euler(u_0: np.array, derivative_matrix: np.array, tau: float, m: int):
    n = len(u_0) - 1
    u = np.empty((m+1, n+1))
    u[0] = u_0
    for i in range(1, m+1):
        u[i] = linear_solver(np.eye(n+1) - tau * derivative_matrix, u[i-1])
    return u


def trapezoids(u_0: np.array, derivative_matrix: np.array, tau: float, m: int):
    n = len(u_0) - 1
    u = np.empty((m+1, n+1))
    u[0] = u_0
    for i in range(1, m+1):
        u[i] = linear_solver(
            np.eye(n+1) - 0.5 * tau * derivative_matrix,
            (np.eye(n + 1) + 0.5 * tau * derivative_matrix) @ u[i-1]
        )
    return u
