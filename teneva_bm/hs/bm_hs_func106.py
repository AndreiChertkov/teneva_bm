import numpy as np
from teneva_bm import Bm


class BmHsFunc106(Bm):
    def __init__(self, d=8, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 106 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 100 | <= 10000
                x[1] | >= 1000 | <= 10000
                x[2] | >= 1000 | <= 10000
                x[3] | >= 10 | <= 1000
                x[4] | >= 10 | <= 1000
                x[5] | >= 10 | <= 1000
                x[6] | >= 10 | <= 1000
                x[7] | >= 10 | <= 1000
            F - objective function
                x[0] + x[1] + x[2]
            C - constraint function
                1 - 0.0025 * (x[3] + x[5]) >= 0
                1 - 0.0025 * (x[4] + x[6] - x[3]) >= 0
                1 - 0.01 * (x[7] - x[4]) >= 0
                x[0] * x[5] - 833.33252 * x[3] - 100 * x[0] + 83333.333 >= 0
                x[1] * x[6] - 1250 * x[4] - x[1] * x[3] + 1250 * x[3] >= 0
                x[2] * x[7] - 1250000 - x[2] * x[4] + 2500 * x[4] >= 0
            The exact global minimum is approx. known:
                y ~= 7049.248
                x[0] ~= 579.307
                x[1] ~= 1359.971
                x[2] ~= 5109.971
                x[3] ~= 182.018
                x[4] ~= 295.601
                x[5] ~= 217.982
                x[6] ~= 286.417
                x[7] ~= 395.601
            Hyperparameters: 
                * The dimension d should be 8
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([100, 1000, 1000, 10, 10, 10, 10, 10], [10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000])
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

    def _constr_batch(self, X):
        c_1 = -1 * (1 - 0.0025 * (X[:, 3] + X[:, 5]))
        c_2 = -1 * (1 - 0.0025 * (X[:, 4] + X[:, 6] - X[:, 3]))
        c_3 = -1 * (1 - 0.01 * (X[:, 7] - X[:, 4]))
        c_4 = -1 * (X[:, 0] * X[:, 5] - 833.33252 * X[:, 3] - 100 * X[:, 0] + 83333.333)
        c_5 = -1 * (X[:, 1] * X[:, 6] - 1250 * X[:, 4] - X[:, 1] * X[:, 3] + 1250 * X[:, 3])
        c_6 = -1 * (X[:, 2] * X[:, 7] - 1250000 - X[:, 2] * X[:, 4] + 2500 * X[:, 4])
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return X[:, 0] + X[:, 1] + X[:, 2]