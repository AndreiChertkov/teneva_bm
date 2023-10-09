import numpy as np
from teneva_bm import Bm


class BmHsFunc107(Bm):
    def __init__(self, d=9, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 107 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                c = (48.4 / 50.176) * sin(0.25)
                d = (48.4 / 50.176) * cos(0.25)
            x - continuous control
                x[0] | >= 0
                x[1] | >= 0
                x[2]
                x[3]
                x[4] | >= 0.90909 | <= 1.0909
                x[5] | >= 0.90909 | <= 1.0909
                x[6] | >= 0.90909 | <= 1.0909
                x[7]
                x[8]
            F - objective function
                3000 * x[0] + 1000 * x[0] ** 3 + 2000 * x[1] + 666.667 * x[1] ** 3
            C - constraint function
                y1 = sin(x[7])
                y2 = cos(x[7])
                y3 = sin(x[8])
                y4 = cos(x[8])
                y5 = sin(x[7] - x[8])
                y6 = cos(x[7] - x[8])
                0.4 - x[0] + 2 * c * x[4] ** 2 - x[4] * x[5] * (d * y1 + c * y2) - x[4] * x[6] * (d * y3 + c * y4) = 0
                0.4 - x[1] + 2 * c * x[5] ** 2 + x[4] * x[5] * (d * y1 - c * y2) + x[5] * x[6] * (d * y5 - c * y6) = 0
                0.8 + 2 * c * x[6] ** 2 + x[4] * x[6] * (d * y3 - c * y4) - x[5] * x[6] * (d * y5 + c * y6) = 0
                0.2 - x[2] + 2 * d * x[4] ** 2 + x[4] * x[5] * (c * y1 - d * y2) + x[4] * x[6] * (c * y3 - d * y4) = 0
                0.2 - x[3] + 2 * d * x[5] ** 2 - x[4] * x[5] * (c * y1 + d * y2) - x[5] * x[6] * (c * y5 + d * y6) = 0
                (-0.337) + 2 * d * x[6] ** 2 - x[4] * x[6] * (c * y3 + d * y4) + x[5] * x[6] * (c * y5 - d * y6) = 0
            The exact global minimum is approx. known:
                y ~= 5055.012
                x[0] ~= 0.667
                x[1] ~= 1.022
                x[2] ~= 0.228
                x[3] ~= 0.185
                x[4] ~= 1.091
                x[5] ~= 1.091
                x[6] ~= 1.069
                x[7] ~= 0.107
                x[8] ~= -0.339
            Hyperparameters: 
                * The dimension d should be 9
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [0, 0, -10, -10, 0.90909, 0.90909, 0.90909, -10, -10], 
            [+10, +10, +10, +10, 1.0909, 1.0909, 1.0909, +10, +10]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 9}

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
        self.parameters = {
            'c': (48.4 / 50.176) * np.sin(0.25),
            'd': (48.4 / 50.176) * np.cos(0.25)
        }

    def _constr_batch(self, X):
        y1 = np.sin(X[:, 7])
        y2 = np.cos(X[:, 7])
        y3 = np.sin(X[:, 8])
        y4 = np.cos(X[:, 8])
        y5 = np.sin(X[:, 7] - X[:, 8])
        y6 = np.cos(X[:, 7] - X[:, 8])
        c_1 = np.abs(
            0.4 - X[:, 0] + 2 * self.parameters['c'] * X[:, 4] ** 2 - 
            X[:, 4] * X[:, 5] * (self.parameters['d'] * y1 + self.parameters['c'] * y2) - 
            X[:, 4] * X[:, 6] * (self.parameters['d'] * y3 + self.parameters['c'] * y4)
        )
        c_2 = np.abs(
            0.4 - X[:, 1] + 2 * self.parameters['c'] * X[:, 5] ** 2 + 
            X[:, 4] * X[:, 5] * (self.parameters['d'] * y1 - self.parameters['c'] * y2) + 
            X[:, 5] * X[:, 6] * (self.parameters['d'] * y5 - self.parameters['c'] * y6)
        )
        c_3 = np.abs(
            0.8 + 2 * self.parameters['c'] * X[:, 6] ** 2 + 
            X[:, 4] * X[:, 6] * (self.parameters['d'] * y3 - self.parameters['c'] * y4) - 
            X[:, 5] * X[:, 6] * (self.parameters['d'] * y5 + self.parameters['c'] * y6)
        )
        c_4 = np.abs(
            0.2 - X[:, 2] + 2 * self.parameters['d'] * X[:, 4] ** 2 + 
            X[:, 4] * X[:, 5] * (self.parameters['c'] * y1 - self.parameters['d'] * y2) + 
            X[:, 4] * X[:, 6] * (self.parameters['c'] * y3 - self.parameters['d'] * y4)
        )
        c_5 = np.abs(
            0.2 - X[:, 3] + 2 * self.parameters['d'] * X[:, 5] ** 2 - 
            X[:, 4] * X[:, 5] * (self.parameters['c'] * y1 + self.parameters['d'] * y2) - 
            X[:, 5] * X[:, 6] * (self.parameters['c'] * y5 + self.parameters['d'] * y6)
        )
        c_6 = np.abs(
            (-0.337) + 2 * self.parameters['d'] * X[:, 6] ** 2 - 
            X[:, 4] * X[:, 6] * (self.parameters['c'] * y3 + self.parameters['d'] * y4) + 
            X[:, 5] * X[:, 6] * (self.parameters['c'] * y5 - self.parameters['d'] * y6)
        )
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 3000 * X[:, 0] + 1000 * X[:, 0] ** 3 + 2000 * X[:, 1] + 666.667 * X[:, 1] ** 3