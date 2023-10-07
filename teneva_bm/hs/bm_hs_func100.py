import numpy as np
from teneva_bm import Bm


class BmHsFunc100(Bm):
    def __init__(self, d=7, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 100 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0]
                x[1]
                x[2]
                x[3]
                x[4]
                x[5]
                x[6]
            F - objective function
                (x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + \
                3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2 + \
                x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
            C - constraint function
                127 - 2 * x[0] ** 2 - 3 * x[1] ** 4 - x[2] - 4 * x[3] ** 2 - 5 * x[4] >= 0
                282 - 7 * x[0] - 3 * x[1] - 10 * x[2] ** 2 - x[3] + x[4] >= 0
                196 - 23 * x[0] - x[1] ** 2 - 6 * x[5] ** 2 + 8 * x[6] >= 0
                (-4) * x[0] ** 2 - x[1] ** 2 + 3 * x[0] * x[1] - 2 * x[2] ** 2 - 5 * x[5] +11 * x[6] >= 0
            The exact global minimum is approx. known:
                y ~= 680.630
                x[0] ~= 2.330
                x[1] ~= 1.951
                x[2] ~= -0.478
                x[3] ~= 4.366
                x[4] ~= -0.624
                x[5] ~= 1.038
                x[6] ~= 1.594
            Hyperparameters: 
                * The dimension d should be 7
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10, -10, -10, -10, -10], [+10, +10, +10, +10, +10, +10, +10])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

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

    def _constr_batch(self, X):
        c_1 = -1 * (127 - 2 * X[:, 0] ** 2 - 3 * X[:, 1] ** 4 - X[:, 2] - 4 * X[:, 3] ** 2 - 5 * X[:, 4])
        c_2 = -1 * (282 - 7 * X[:, 0] - 3 * X[:, 1] - 10 * X[:, 2] ** 2 - X[:, 3] + X[:, 4])
        c_3 = -1 * (196 - 23 * X[:, 0] - X[:, 1] ** 2 - 6 * X[:, 5] ** 2 + 8 * X[:, 6])
        c_4 = -1 * ((-4) * X[:, 0] ** 2 - X[:, 1] ** 2 + 3 * X[:, 0] * X[:, 1] - 2 * X[:, 2] ** 2 - 5 * X[:, 5] +11 * X[:, 6])
        return np.array([c_1, c_2, c_3, c_4])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (X[:, 0] - 10) ** 2 + 5 * (X[:, 1] - 12) ** 2 + \
               X[:, 2] ** 4 + 3 * (X[:, 3] - 11) ** 2 + 10 * X[:, 4] ** 6 + \
               7 * X[:, 5] ** 2 + X[:, 6] ** 4 - 4 * X[:, 5] * X[:, 6] - \
               10 * X[:, 5] - 8 * X[:, 6]