import numpy as np

from typing import Callable, Tuple, List
from scipy.integrate import quad, cumtrapz

from rates import make_b, make_sigma2
from params import Parameters


def make_kappa(b: Callable[[float], float], sigma2: Callable[[float], float])\
        -> Callable[[float, float], float]:
    """"Dla zadanych funkcji b i sigma2 zwraca funkcję kappa.

    :param b:
    :param sigma2:
    :return:
    """
    def kappa(start: float, end: float) -> float:
        def func(n):
            return 2*b(n)/sigma2(n)
        int_func, *_ = quad(func, start, end)
        return int_func
    return kappa


def compute_integral(step: float, n: int, params: Parameters) -> Tuple[np.ndarray, np.ndarray]:
    """Liczy wartość oczekiwaną momentu zatrzymania.

    :param step: krok całkowania
    :param n: liczba kroków
    :param params: parametry modelu
    :return:
    """
    b = make_b(params)
    sigma2 = make_sigma2(params)
    kappa = make_kappa(b, sigma2)
    # Trick, żeby dostać zaokrąglonego inta od razu.
    # Epsilon = h
    #n_of_steps = int(0.5 + (params.n_star - step) / step)
    linspace = np.linspace(step, params.n_star, n)
    kappa_pieces = np.array([0] + [kappa(linspace[i], linspace[i+1]) for i in range(n-1)])
    psi_vals = np.exp(np.cumsum(kappa_pieces))

    eta_integrand_vals = psi_vals/sigma2(linspace)
    eta_vals = cumtrapz(eta_integrand_vals, linspace, initial=0)

    tau_vals = 2*np.flipud(cumtrapz(eta_vals/psi_vals, linspace, initial=0))
    return linspace, tau_vals
