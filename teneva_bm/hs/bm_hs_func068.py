import numpy as np
from teneva_bm import Bm
from scipy.special import erfc


class BmHsFunc068(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 068 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .-------------------------------------.
            | F(x | p, t) -> min s.t. C(x) = True |
            .-------------------------------------.
            p - parameters
                a = 0.0001
                b = 1
                d = 1
                n = 24
            x - continuous control
                x[0] | >= 0.0001 | <= 100
                x[1] | >= 0      | <= 100
                x[2] | >= 0      | <= 2
                x[3] | >= 0      | <= 2
            t - intermediates
                argn = (-1) * x[1] - d * sqrt(n)
                arg0 = (-1) * x[1]
                argp = (-1) * x[1] + d * sqrt(n)
                phin = (1 / 2) * erfc((-1) * argn / sqrt(2))
                phi0 = (1 / 2) * erfc((-1) * arg0 / sqrt(2))
                phip = (1 / 2) * erfc((-1) * argp / sqrt(2))
                num = b * (exp(x[0]) - 1) - x[2]
                den = exp(x[0]) - 1 + x[3]
            F - objective function
                (a * n - (num / den) * x[3]) / x[0]
            C - constraint function
                x[2] - 2 * phi0 = 0
                x[3] - phip - phin = 0
            The exact global minimum is approx. known:
                y ~= -0.920
                x[0] ~= 0.068
                x[1] ~= 3.646
                x[2] ~= 0.000
                x[3] ~= 0.895
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0.0001, 0, 0, 0], [100, 100, 2, 2])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 4}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def with_constr(self):
        return True

    def set_parameters(self):
        self.parameters = {'a': 0.0001, 'b': 1, 'd': 1, 'n': 24}
    
    def intermediates(self, X):
        argn = (-1) * X[:, 1] - self.parameters['d'] * np.sqrt(self.parameters['n'])
        arg0 = (-1) * X[:, 1]
        argp = (-1) * X[:, 1] + self.parameters['d'] * np.sqrt(self.parameters['n'])
        phin = (1 / 2) * erfc((-1) * argn / np.sqrt(2))
        phi0 = (1 / 2) * erfc((-1) * arg0 / np.sqrt(2))
        phip = (1 / 2) * erfc((-1) * argp / np.sqrt(2))
        return phin, phi0, phip
    
    def _constr_batch(self, X):
        phin, phi0, phip = self.intermediates(X)
        c_1 = np.abs(X[:, 2] - 2 * phi0)
        c_2 = np.abs(X[:, 3] - phip - phin)
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c
    
    def target_batch(self, X):
        num = self.parameters['b'] * (np.exp(X[:, 0]) - 1) - X[:, 2]
        den = np.exp(X[:, 0]) - 1 + X[:, 3]
        obj = (self.parameters['a'] * self.parameters['n'] - (num / den) * X[:, 3]) / X[:, 0]
        return obj