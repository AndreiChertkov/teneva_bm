import numpy as np
from teneva_bm import Bm


class BmHsFunc081(Bm):
    def __init__(self, d=5, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 081 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= -2.3 | <= 2.3
                x[1] | >= -2.3 | <= 2.3
                x[2] | >= -3.2 | <= 3.2
                x[3] | >= -3.2 | <= 3.2
                x[4] | >= -3.2 | <= 3.2
            F - objective function
                exp(x[0] * x[1] * x[2] * x[3] * x[4]) - (1 / 2) * (x[0] ** 3 + x[1] ** 3 + 1) ** 2
            C - constraint function
                x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 - 10 = 0
                x[1] * x[2] - 5 * x[3] * x[4] = 0
                x[0] ** 3 + x[1] ** 3 + 1 = 0
            The exact global minimum is approx. known:
                y ~= 0.054
                x[0] ~= -1.717
                x[1] ~= 1.596
                x[2] ~= 1.827
                x[3] ~= -0.764
                x[4] ~= -0.764
            Hyperparameters: 
                * The dimension d should be 5
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-2.3, -2.3, -3.2, -3.2, -3.2], [2.3, 2.3, 3.2, 3.2, 3.2])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 5}

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
        c_1 = np.abs(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2 + X[:, 4] ** 2 - 10)
        c_2 = np.abs(X[:, 1] * X[:, 2] - 5 * X[:, 3] * X[:, 4])
        c_3 = np.abs(X[:, 0] ** 3 + X[:, 1] ** 3 + 1)
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return np.exp(X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3] * X[:, 4]) - (1 / 2) * (X[:, 0] ** 3 + X[:, 1] ** 3 + 1) ** 2