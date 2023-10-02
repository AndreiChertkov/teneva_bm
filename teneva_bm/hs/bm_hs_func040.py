import numpy as np
from teneva_bm import Bm


class BmHsFunc040(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 040 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0]
                x[1]
                x[2]
                x[3]
            F - objective function
                -x[0] * x[1] * x[2] * x[3]
            C - constraint function
                x[0] ** 3 + x[1] ** 2 - 1 = 0
                x[0] ** 2 * x[3] - x[2] = 0
                x[3] ** 2 - x[1] = 0
            The exact global minimum is known:
                y = -0.25
                x[0] = 2 ** (-1/3) 
                x[1] = 2 ** (-1/2) 
                x[2] = 2 ** (-11/12) 
                x[3] = 2 ** (-1/4) 
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10, -10], [+10, +10, +10, +10])
        self.set_min(x=[2 ** (-1/3), 2 ** (-1/2), 2 ** (-11/12), 2 ** (-1/4)], y=-0.25)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

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

    def _constr_batch(self, X):
        c_1 = np.abs(X[:, 0] ** 3 + X[:, 1] ** 2 - 1)
        c_2 = np.abs(X[:, 0] ** 2 * X[:, 3] - X[:, 2])
        c_3 = np.abs(X[:, 3] ** 2 - X[:, 1])
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return -X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]