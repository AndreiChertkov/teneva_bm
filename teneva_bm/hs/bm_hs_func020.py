import numpy as np
from teneva_bm import Bm


class BmHsFunc020(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 020 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= -1/2 | <= 1/2
                x[1]
            F - objective function
                100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
            C - constraint function
                x[0] + x[1] ** 2 >= 0
                x[0] ** 2 + x[1] >= 0
                x[0] ** 2 + x[1] ** 2 - 1 >= 0
            The exact global minimum is known:
                y = 81.5 - 25 * sqrt(3)
                x[0] = 0.5
                x[1] = sqrt(3)/2
            Hyperparameters: 
                * The dimension d should be 2
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-0.5, -10], [0.5, +10])
        self.set_min(x=[0.5, np.sqrt(3)/2], y=81.5 - 25 * np.sqrt(3))
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 2}

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
        c_1 = -1 * (X[:, 0] + X[:, 1] ** 2)
        c_2 = -1 * (X[:, 0] ** 2 + X[:, 1])
        c_3 = -1 * (X[:, 0] ** 2 + X[:, 1] ** 2 - 1)
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 100 * (X[:, 1] - X[:, 0] ** 2) ** 2 + (1 - X[:, 0]) ** 2