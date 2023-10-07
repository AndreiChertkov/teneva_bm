import numpy as np
from teneva_bm import Bm


class BmHsFunc074(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 074 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0 | <= 1200
                x[1] | >= 0 | <= 1200
                x[2] | >= -0.55 | <= 0.55
                x[3] | >= -0.55 | <= 0.55
            F - objective function
                3 * x[0] + 1.0e-6 * x[0] ** 3 + 2 * x[1] + 2.0e-6 * x[1] ** 3 / 3
            C - constraint function
                x[3] - x[2] + 0.55 >= 0
                x[2] - x[3] + 0.55 >= 0
                1000 * sin((-1) * x[2] - 0.25) + 1000 * sin((-1) * x[3] - 0.25) + 894.8 - x[0] = 0
                1000 * sin(x[2] - 0.25) + 1000 * sin(x[2] - x[3] - 0.25) + 894.8 - x[1] = 0
                1000 * sin(x[3] - 0.25) + 1000 * sin(x[3] - x[2] - 0.25) + 1294.8 = 0
            The exact global minimum is approx. known:
                y ~= 5126.498
                x[0] ~= 679.945
                x[1] ~= 1026.067
                x[2] ~= 0.119
                x[3] ~= -0.396
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, -0.55, -0.55], [1200, 1200, 0.55, 0.55])
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
        c_1 = -1 * (X[:, 3] - X[:, 2] + 0.55)
        c_2 = -1 * (X[:, 2] - X[:, 3] + 0.55)
        c_3 = np.abs(1000 * np.sin((-1) * X[:, 2] - 0.25) + 1000 * np.sin((-1) * X[:, 3] - 0.25) + 894.8 - X[:, 0])
        c_4 = np.abs(1000 * np.sin(X[:, 2] - 0.25) + 1000 * np.sin(X[:, 2] - X[:, 3] - 0.25) + 894.8 - X[:, 1])
        c_5 = np.abs(1000 * np.sin(X[:, 3] - 0.25) + 1000 * np.sin(X[:, 3] - X[:, 2] - 0.25) + 1294.8)
        return np.array([c_1, c_2, c_3, c_4, c_5])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 3 * X[:, 0] + 1.0e-6 * X[:, 0] ** 3 + 2 * X[:, 1] + 2.0e-6 * X[:, 1] ** 3 / 3