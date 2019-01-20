class Parameters(object):
    # Parametry trzeba podawac jako keyword
    def __init__(self, *, c, r, d, alpha, a, u, gamma):
        self.c = c
        self.r = r
        self.d = d
        self.alpha = alpha
        self.a = a
        self.u = u
        self.gamma = gamma

        self.n_star = u/gamma


ex_params_1 = Parameters(
    c=5,
    r=5,
    d=4,
    alpha=0.1,
    u=6,
    gamma=0.3,
    a=0.55
)