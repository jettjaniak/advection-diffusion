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


def compute_psi_vals(linspace, params: Parameters):
    """Liczy wartość oczekiwaną momentu zatrzymania.

    :param linspace:
    :param params: parametry modelu
    :return:
    """
    n = len(linspace)
    b = make_b(params)
    sigma2 = make_sigma2(params)
    kappa = make_kappa(b, sigma2)
    kappa_pieces = np.array([0] + [kappa(linspace[i], linspace[i+1]) for i in range(n-1)])
    kappa_vals = np.cumsum(kappa_pieces)
    return np.exp(kappa_vals)


def compute_eta_vals(linspace, psi_vals, sigma2):
    """Liczy wartość oczekiwaną momentu zatrzymania.

    :param linspace:
    :param params: parametry modelu
    :return:
    """
    eta_integrand_vals = psi_vals / sigma2(linspace)
    return cumtrapz(eta_integrand_vals, linspace, initial=0)


def compute_integral(step: float, n: int, params: Parameters) -> Tuple[np.ndarray, np.ndarray]:
    """Liczy wartość oczekiwaną momentu zatrzymania.

    :param step: krok całkowania
    :param n: liczba kroków
    :param params: parametry modelu
    :return:
    """
    sigma2 = make_sigma2(params)

    linspace = np.linspace(step, params.n_star, n)
    psi_vals = compute_psi_vals(linspace, params)

    eta_vals = compute_eta_vals(linspace, psi_vals, sigma2)
    pre_tau_vals = cumtrapz(eta_vals/psi_vals, linspace, initial=0)
    half_tau_vals_reversed = pre_tau_vals[-1] - np.flipud(pre_tau_vals)
    tau_vals = 2*np.flipud(half_tau_vals_reversed)
    return linspace, tau_vals
