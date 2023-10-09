import numpy as np
from teneva_bm import Bm


class BmHsFunc104(Bm):
    def __init__(self, d=8, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 104 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | t) -> min s.t. C(x) = True |
            .----------------------------------.
            x - continuous control
                x[0] | >= 0.1 | <= 10
                x[1] | >= 0.1 | <= 10
                x[2] | >= 0.1 | <= 10
                x[3] | >= 0.1 | <= 10
                x[4] | >= 0.1 | <= 10
                x[5] | >= 0.1 | <= 10
                x[6] | >= 0.1 | <= 10
                x[7] | >= 0.1 | <= 10
            t - intermediates
                t = .4 * (x[0] / x[6]) ** .67 + .4 * (x[1] / x[7]) ** .67 + 10 - x[0] - x[1]
            F - objective function
                t
            C - constraint function
                1 - 0.0588 * x[4] * x[6] - 0.1 * x[0] >= 0
                1 - 0.0588 * x[5] * x[7] - 0.1 * x[0] - 0.1 * x[1] >= 0
                1 - 4 * x[2] / x[4] - 2 / (x[2] ** 0.71 * x[4]) - 0.0588 * x[6] / x[2] ** 1.3 >= 0
                1 - 4 * x[3] / x[5] - 2 / (x[3] ** 0.71 * x[5]) - 0.0588 * x[7] / x[3] ** 1.3 >= 0
                t - 1 >= 0
                4.2 - t >= 0
            The exact global minimum is approx. known:
                y ~= 3.951
                x[0] ~= 6.465
                x[1] ~= 2.233
                x[2] ~= 0.667
                x[3] ~= 0.596
                x[4] ~= 5.933
                x[5] ~= 5.527
                x[6] ~= 1.013
                x[7] ~= 0.401
            Hyperparameters: 
                * The dimension d should be 8
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [10, 10, 10, 10, 10, 10, 10, 10])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 8}

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
        t = 0.4 * (X[:, 0] / X[:, 6]) ** 0.67 + 0.4 * (X[:, 1] / X[:, 7]) ** 0.67 + 10 - X[:, 0] - X[:, 1]
        return t

    def _constr_batch(self, X):
        t = self.intermediates(X)
        c_1 = -1 * (1 - 0.0588 * X[:, 4] * X[:, 6] - 0.1 * X[:, 0])
        c_2 = -1 * (1 - 0.0588 * X[:, 5] * X[:, 7] - 0.1 * X[:, 0] - 0.1 * X[:, 1])
        c_3 = -1 * (1 - 4 * X[:, 2] / X[:, 4] - 2 / (X[:, 2] ** 0.71 * X[:, 4]) - 0.0588 * X[:, 6] / X[:, 2] ** 1.3)
        c_4 = -1 * (1 - 4 * X[:, 3] / X[:, 5] - 2 / (X[:, 3] ** 0.71 * X[:, 5]) - 0.0588 * X[:, 7] / X[:, 3] ** 1.3)
        c_5 = -1 * (t - 1)
        c_6 = -1 * (4.2 - t)
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        t = self.intermediates(X)
        return t