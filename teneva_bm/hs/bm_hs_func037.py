import numpy as np
from teneva_bm import Bm


class BmHsFunc037(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 037 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0 | <= 42
                x[1] | >= 0 | <= 42
                x[2] | >= 0 | <= 42
            F - objective function
                -x[0] * x[1] * x[2]
            C - constraint function
                72 - x[0] - 2 * x[1] - 2 * x[2] >= 0
                x[0] + 2 * x[1] + 2 * x[2] >= 0
            The exact global minimum is known:
                y = -3456
                x[0] = 24
                x[1] = 12
                x[2] = 12
            Hyperparameters: 
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0], [42, 42, 42])
        self.set_min(x=[24, 12, 12], y=-3456)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 3}

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
        c_1 = -1 * (72 - X[:, 0] - 2 * X[:, 1] - 2 * X[:, 2])
        c_2 = -1 * (X[:, 0] + 2 * X[:, 1] + 2 * X[:, 2])
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return -X[:, 0] * X[:, 1] * X[:, 2]