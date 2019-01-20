import numpy as np

from typing import Callable

from params import Parameters


def c(n: float, params: Parameters) -> float:
    """Tempo tworzenia wiązań.

    :param n: liczba wiązań
    :param params: parametry modelu
    """
    return params.c


def r(n: float, params: Parameters) -> float:
    """Tempo namnażania wiązań.

    :param n: liczba wiązań
    :param params: parametry modelu
    """
    return params.r


def d(n: float, params: Parameters) -> float:
    """Tempo rozpadu wiązań.

    :param n: liczba wiązań
    :param params: parametry modelu
    """
    return params.d*np.exp(params.alpha*(params.u-params.gamma*n))


def make_b(params: Parameters) -> Callable[[float], float]:
    """Dla zadanych parametrów modelu zwraca funkcję b.

    :param params: parametry modelu
    :return: funkcja b
    """
    def b(n: float) -> float:
        return c(n, params) + (r(n, params)-d(n,params))*n
    return b


def make_sigma2(params: Parameters) -> Callable[[float], float]:
    """Dla zadanych parametrów modelu zwraca kwadrat funkcji sigma.

    :param params: parametry modelu
    :return: funkcja sigma^2
    """
    def sigma2(n: float) -> float:
        return 2*params.a*n
    return sigma2
