import numpy as np
from teneva_bm import Bm


class BmHsFunc113(Bm):
    def __init__(self, d=10, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 113 from the Hock & Schittkowski collection.
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
                x[7]
                x[8]
                x[9]
            F - objective function
                x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + 
                (x[2] - 10) ** 2 + 4 * (x[3] - 5) ** 2 + (x[4] - 3) ** 2 + 2 * (x[5] - 1) ** 2 + 
                5 * x[6] ** 2 + 7 * (x[7] - 11) ** 2 + 2 * (x[8] - 10) ** 2 + (x[9] - 7) ** 2 + 45
            C - constraint function
                105 - 4 * x[0] - 5 * x[1] + 3 * x[6] - 9 * x[7] >= 0
                (-1) * 10 * x[0] + 8 * x[1] + 17 * x[6] - 2 * x[7] >= 0
                8 * x[0] - 2 * x[1] - 5 * x[8] + 2 * x[9] + 12 >= 0
                (-3) * (x[0] - 2) ** 2 - 4 * (x[1] - 3) ** 2 - 2 * x[2] ** 2 + 7 * x[3] + 120 >= 0
                (-5) * x[0] ** 2 - 8 * x[1] - (x[2] - 6) ** 2 + 2 * x[3] + 40 >= 0
                (-1 / 2) * (x[0] - 8) ** 2 - 2 * (x[1] - 4) ** 2 - 3 * x[4] ** 2 + x[5] + 30 >= 0
                (-1) * x[0] ** 2 - 2 * (x[1] - 2) ** 2 + 2 * x[0] * x[1] - 14 * x[4] + 6 * x[5] >= 0
                3 * x[0] - 6 * x[1] - 12 * (x[8] - 8) ** 2 + 7 * x[9] >= 0
            The exact global minimum is approx. known:
                y ~= 24.306
                x[0] ~= 2.172
                x[1] ~= 2.364
                x[2] ~= 8.774
                x[3] ~= 5.096
                x[4] ~= 0.991
                x[5] ~= 1.431
                x[6] ~= 1.322
                x[7] ~= 9.829
                x[8] ~= 8.280
                x[9] ~= 8.376
            Hyperparameters: 
                * The dimension d should be 10
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10], 
            [+10, +10, +10, +10, +10, +10, +10, +10, +10, +10]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 10}

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
        c_1 = -1 * (105 - 4 * X[:, 0] - 5 * X[:, 1] + 3 * X[:, 6] - 9 * X[:, 7])
        c_2 = -1 * ((-1) * 10 * X[:, 0] + 8 * X[:, 1] + 17 * X[:, 6] - 2 * X[:, 7])
        c_3 = -1 * (8 * X[:, 0] - 2 * X[:, 1] - 5 * X[:, 8] + 2 * X[:, 9] + 12)
        c_4 = -1 * ((-3) * (X[:, 0] - 2) ** 2 - 4 * (X[:, 1] - 3) ** 2 - 2 * X[:, 2] ** 2 + 7 * X[:, 3] + 120)
        c_5 = -1 * ((-5) * X[:, 0] ** 2 - 8 * X[:, 1] - (X[:, 2] - 6) ** 2 + 2 * X[:, 3] + 40)
        c_6 = -1 * ((-1 / 2) * (X[:, 0] - 8) ** 2 - 2 * (X[:, 1] - 4) ** 2 - 3 * X[:, 4] ** 2 + X[:, 5] + 30)
        c_7 = -1 * ((-1) * X[:, 0] ** 2 - 2 * (X[:, 1] - 2) ** 2 + 2 * X[:, 0] * X[:, 1] - 14 * X[:, 4] + 6 * X[:, 5])
        c_8 = -1 * (3 * X[:, 0] - 6 * X[:, 1] - 12 * (X[:, 8] - 8) ** 2 + 7 * X[:, 9])
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 0] * X[:, 1] - 14 * X[:, 0] - \
               16 * X[:, 1] + (X[:, 2] - 10) ** 2 + 4 * (X[:, 3] - 5) ** 2 + (X[:, 4] - 3) ** 2 + \
               2 * (X[:, 5] - 1) ** 2 + 5 * X[:, 6] ** 2 + 7 * (X[:, 7] - 11) ** 2 + \
               2 * (X[:, 8] - 10) ** 2 + (X[:, 9] - 7) ** 2 + 45