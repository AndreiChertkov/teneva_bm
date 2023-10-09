import numpy as np
from teneva_bm import Bm


class BmHsFunc114(Bm):
    def __init__(self, d=10, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 114 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a = .99
                b = .9
            x - continuous control
                x[0] | >= .00001 | <= 2000
                x[1] | >= .00001 | <= 16000
                x[2] | >= .00001 | <= 120
                x[3] | >= .00001 | <= 5000
                x[4] | >= .00001 | <= 2000
                x[5] | >= 85 | <= 93
                x[6] | >= 90 | <= 95
                x[7] | >= 3 | <= 12
                x[8] | >= 1.2 | <= 4
                x[9] | >= 145 | <= 162
            F - objective function
                5.04 * x[0] + .035 * x[1] + 10 * x[2] + 3.36 * x[4] - .063 * x[3] * x[6]
            C - constraint function
                35.82 - .222 * x[9] - b * x[8] >= 0
                (-133) + 3 * x[6] - a * x[9] >= 0
                (-1) * g[0] + x[8] * (1 / b - b) >= 0
                (-1) * g[1] + (1 / a - a) * x[9] >= 0
                1.12 * x[0] + .13167 * x[0] * x[7] - .00667 * x[0] * x[7] ** 2 - a * x[3] >= 0
                57.425 + 1.098 * x[7] - .038 * x[7] ** 2 + .325 * x[5] - a * x[6] >= 0
                (-1) * g[4] + (1 / a - a) * x[3] >= 0
                (-1) * g[5] + (1 / a - a) * x[6] >= 0
                1.22 * x[3] - x[0] - x[4] = 0
                98000 * x[2] / (x[3] * x[8] + 1000 * x[2]) - x[5] = 0
                (x[1] + x[4]) / x[0] - x[7] = 0
            The exact global minimum is approx. known:
                y ~= -1768.807
                x[0] ~= 1698.095
                x[1] ~= 15818.615
                x[2] ~= 54.103
                x[3] ~= 3031.225
                x[4] ~= 2000
                x[5] ~= 90.115
                x[6] ~= 95
                x[7] ~= 10.493
                x[8] ~= 1.562
                x[9] ~= 153.535
            Hyperparameters: 
                * The dimension d should be 10
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 85, 90, 3, 1.2, 145], 
            [2000, 16000, 120, 5000, 2000, 93, 95, 12, 4, 162]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

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

    def set_parameters(self):
        self.parameters = {'a': 0.99, 'b': 0.9,}

    def _constr_batch(self, X):
        c_1 = -1 * (35.82 - .222 * X[:, 9] - self.parameters['b'] * X[:, 8])
        c_2 = -1 * ((-133) + 3 * X[:, 6] - self.parameters['a'] * X[:, 9])
        c_3 = -1 * (c_1 + X[:, 8] * (1 / self.parameters['b'] - self.parameters['b']))
        c_4 = -1 * (c_2 + (1 / self.parameters['a'] - self.parameters['a']) * X[:, 9])
        c_5 = -1 * (1.12 * X[:, 0] + .13167 * X[:, 0] * X[:, 7] - .00667 * X[:, 0] * X[:, 7] ** 2 - \
                    self.parameters['a'] * X[:, 3])
        c_6 = -1 * (57.425 + 1.098 * X[:, 7] - .038 * X[:, 7] ** 2 + .325 * X[:, 5] - self.parameters['a'] * X[:, 6])
        c_7 = -1 * (c_3 + (1 / self.parameters['a'] - self.parameters['a']) * X[:, 3])
        c_8 = -1 * (c_4 + (1 / self.parameters['a'] - self.parameters['a']) * X[:, 6])
        c_9 = np.abs(1.22 * X[:, 3] - X[:, 0] - X[:, 4])
        c_10 = np.abs(98000 * X[:, 2] / (X[:, 3] * X[:, 8] + 1000 * X[:, 2]) - X[:, 5])
        c_11 = np.abs((X[:, 1] + X[:, 4]) / X[:, 0] - X[:, 7])
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 5.04 * X[:, 0] + .035 * X[:, 1] + 10 * X[:, 2] + 3.36 * X[:, 4] - .063 * X[:, 3] * X[:, 6]