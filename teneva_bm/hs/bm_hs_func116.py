import numpy as np
from teneva_bm import Bm


class BmHsFunc116(Bm):
    def __init__(self, d=13, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 116 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | t) -> min s.t. C(x) = True |
            .----------------------------------.
            x - continuous control
                x[0] | >= 0.1 | <= 1
                x[1] | >= 0.1 | <= 1
                x[2] | >= 0.1 | <= 1
                x[3] | >= 0.0001 | <= 0.1
                x[4] | >= 0.1 | <= 0.9
                x[5] | >= 0.1 | <= 0.9
                x[6] | >= 0.1 | <= 1000
                x[7] | >= 0.1 | <= 1000
                x[8] | >= 500 | <= 1000
                x[9] | >= 0.1 | <= 500
                x[10] | >= 1 | <= 150
                x[11] | >= 0.0001 | <= 150
                x[12] | >= 0.0001 | <= 150
            t - intermediates
                t = x[10] + x[11] + x[12]
            F - objective function
                t
            C - constraint function
                x[2] - x[1] >= 0
                x[1] - x[0] >= 0
                1 - .002 * x[6] + .002 * x[7] >= 0
                t - 50 >= 0
                250 - t >= 0
                x[12] - 1.262626 * x[9] + 1.231059 * x[2] * x[9] >= 0
                x[4] - .03475 * x[1] - .975 * x[1] * x[4] + .00975 * x[1] ** 2 >= 0
                x[5] - .03475 * x[2] - .975 * x[2] * x[5] + .00975 * x[2] ** 2 >= 0
                x[4] * x[6] - x[0] * x[7] - x[3] * x[6] + x[3] * x[7] >= 0
                1 - .002 * (x[1] * x[8] + x[4] * x[7] - x[0] * x[7] - x[5] * x[8]) - x[4] - x[5] >= 0
                x[1] * x[8] - x[2] * x[9] - x[5] * x[8] - 500 * x[1] + 500 * x[5] + x[1] * x[9] >= 0
                x[1] - .9 - .002 * (x[1] * x[9] - x[2] * x[9]) >= 0
                x[3] - .03475 * x[0] - .975 * x[0] * x[3] + .00975 * x[0] ** 2 >= 0
                x[10] - 1.262626 * x[7] + 1.231059 * x[0] * x[7] >= 0
                x[11] - 1.262626 * x[8] + 1.231059 * x[1] * x[8] >= 0
            The exact global minimum is approx. known:
                y ~= 97.588
                x[0] ~= 0.804
                x[1] ~= 0.900
                x[2] ~= 0.971
                x[3] ~= 0.100
                x[4] ~= 0.191
                x[5] ~= 0.461
                x[6] ~= 574.078
                x[7] ~= 74.078
                x[8] ~= 500.016
                x[9] ~= 0.100
                x[10] ~= 20.233
                x[11] ~= 77.348
                x[12] ~= 0.007
            Hyperparameters: 
                * The dimension d should be 13
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1, 0.1, 500, 0.1, 1, 0.0001, 0.0001], 
            [1, 1, 1, 0.1, 0.9, 0.9, 1000, 1000, 1000, 500, 150, 150, 150]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 13}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def with_constr(self):
        return True

    def intermediates(self, X):
        t = X[:, 10] + X[:, 11] + X[:, 12]
        return t

    def _constr_batch(self, X):
        t = self.intermediates(X)
        c_1 = -1 * (X[:, 2] - X[:, 1])
        c_2 = -1 * (X[:, 1] - X[:, 0])
        c_3 = -1 * (1 - .002 * X[:, 6] + .002 * X[:, 7])
        c_4 = -1 * (t - 50)
        c_5 = -1 * (250 - t)
        c_6 = -1 * (X[:, 12] - 1.262626 * X[:, 9] + 1.231059 * X[:, 2] * X[:, 9])
        c_7 = -1 * (X[:, 4] - .03475 * X[:, 1] - .975 * X[:, 1] * X[:, 4] + .00975 * X[:, 1] ** 2)
        c_8 = -1 * (X[:, 5] - .03475 * X[:, 2] - .975 * X[:, 2] * X[:, 5] + .00975 * X[:, 2] ** 2)
        c_9 = -1 * (X[:, 4] * X[:, 6] - X[:, 0] * X[:, 7] - X[:, 3] * X[:, 6] + X[:, 3] * X[:, 7])
        c_10 = -1 * (1 - .002 * (X[:, 1] * X[:, 8] + X[:, 4] * X[:, 7] - X[:, 0] * X[:, 7] - \
                    X[:, 5] * X[:, 8]) - X[:, 4] - X[:, 5])
        c_11 = -1 * (X[:, 1] * X[:, 8] - X[:, 2] * X[:, 9] - X[:, 5] * X[:, 8] - 500 * X[:, 1] + \
                    500 * X[:, 5] + X[:, 1] * X[:, 9])
        c_12 = -1 * (X[:, 1] - .9 - .002 * (X[:, 1] * X[:, 9] - X[:, 2] * X[:, 9]))
        c_13 = -1 * (X[:, 3] - .03475 * X[:, 0] - .975 * X[:, 0] * X[:, 3] + .00975 * X[:, 0] ** 2)
        c_14 = -1 * (X[:, 10] - 1.262626 * X[:, 7] + 1.231059 * X[:, 0] * X[:, 7])
        c_15 = -1 * (X[:, 11] - 1.262626 * X[:, 8] + 1.231059 * X[:, 1] * X[:, 8])
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        t = self.intermediates(X)
        return t