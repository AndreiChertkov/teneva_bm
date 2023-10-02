import numpy as np
from teneva_bm import Bm


class BmHsFunc055(Bm):
    def __init__(self, d=6, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 055 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0 | <= 1
                x[1] | >= 0
                x[2] | >= 0
                x[3] | >= 0 | <= 1
                x[4] | >= 0
                x[5] | >= 0
            F - objective function
                x[0] + 2 * x[1] + 4 * x[4] + exp(x[0] * x[3])
            C - constraint function
                x[0] + 2 * x[1] + 5 * x[4] - 6 = 0
                x[0] + x[1] + x[2] - 3 = 0
                x[3] + x[4] + x[5] - 2 = 0
                x[0] + x[3] - 1 = 0
                x[1] + x[4] - 2 = 0
                x[2] + x[5] - 2 = 0
            The exact global minimum is known:
                y = 19/3 
                x[0] = 0
                x[1] = 4/3 
                x[2] = 5/3 
                x[3] = 1
                x[4] = 2/3 
                x[5] = 1/3 
            Hyperparameters: 
                * The dimension d should be 6
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0, 0, 0, 0], [1, +10, +10, 1, +10, +10])
        self.set_min(x=[0, 4/3, 5/3, 1, 2/3, 1/3], y=19/3)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 6}

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
        c_1 = np.abs(X[:, 0] + 2 * X[:, 1] + 5 * X[:, 4] - 6)
        c_2 = np.abs(X[:, 0] + X[:, 1] + X[:, 2] - 3)
        c_3 = np.abs(X[:, 3] + X[:, 4] + X[:, 5] - 2)
        c_4 = np.abs(X[:, 0] + X[:, 3] - 1)
        c_5 = np.abs(X[:, 1] + X[:, 4] - 2)
        c_6 = np.abs(X[:, 2] + X[:, 5] - 2)
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return X[:, 0] + 2 * X[:, 1] + 4 * X[:, 4] + np.exp(X[:, 0] * X[:, 3])