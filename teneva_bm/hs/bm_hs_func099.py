import numpy as np
from teneva_bm import Bm


class BmHsFunc099(Bm):
    def __init__(self, d=7, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 099 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a = [50, 50, 75, 75, 75, 100, 100]
                b = 32
                t = [25, 25, 50, 50, 50, 90, 90]
            x - continuous control
                x[0] | >= 0 | <= 1.58
                x[1] | >= 0 | <= 1.58
                x[2] | >= 0 | <= 1.58
                x[3] | >= 0 | <= 1.58
                x[4] | >= 0 | <= 1.58
                x[5] | >= 0 | <= 1.58
                x[6] | >= 0 | <= 1.58
            F - objective function
                -sum(a * t * cos(x)) ** 2
            C - constraint function
                taxb = t * (a * sin(x) - b)
                sum(taxb) - 1000 = 0
                1.5 * sum(t * taxb) - 100000 = 0
            The exact global minimum is approx. known:
                y ~= -831079891.510
                x[0] ~= 0.542
                x[1] ~= 0.529
                x[2] ~= 0.508
                x[3] ~= 0.480
                x[4] ~= 0.451
                x[5] ~= 0.409
                x[6] ~= 0.353
            Hyperparameters: 
                * The dimension d should be 7
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0, 0, 0, 0, 0], [1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 7}

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
        self.parameters = {
            'a': np.array([50, 50, 75, 75, 7, 10, 100]),
            'b': 32,
            't': np.array([25, 25, 50, 50, 50, 90, 90])
        }

    def _constr_batch(self, X):
        taxb = self.parameters['t'] * (self.parameters['a'] * np.sin(X) - self.parameters['b'])
        c_1 = np.abs(taxb.sum(-1) - 1000)
        c_2 = np.abs(1.5 * (self.parameters['t'] * taxb).sum(-1) - 100000)
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return -sum(self.parameters['t'] * self.parameters['a'] * np.cos(X)) ** 2