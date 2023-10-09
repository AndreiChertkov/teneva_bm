import numpy as np
from teneva_bm import Bm


class BmHsFunc109(Bm):
    def __init__(self, d=9, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 109 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a = 50.176
                b = sin(0.25)
                c = cos(0.25)
            x - continuous control
                x[0] | >= 0
                x[1] | >= 0
                x[2] | >= -0.55 | <= 0.55
                x[3] | >= -0.55 | <= 0.55
                x[4] | >= 196 | <= 252
                x[5] | >= 196 | <= 252
                x[6] | >= 196 | <= 252
                x[7] | >= -400 | <= 800
                x[8] | >= -400 | <= 800
            F - objective function
                3 * x[0] + 1.0e-6 * x[0] ** 3 + 2 * x[1] + 0.522074e-6 * x[1] ** 3
            C - constraint function
                x[3] - x[2] + .55 >= 0
                x[2] - x[3] + .55 >= 0
                2250000 - x[0] ** 2 - x[7] ** 2 >= 0
                2250000 - x[1] ** 2 - x[8] ** 2 >= 0
                ------------------------------------------
                x[4] * x[5] * sin((-1) * x[2] - 1 / 4) +
                x[4] * x[6] * sin((-1) * x[3] - 1 / 4) +
                2 * b * x[4] ** 2 - a * x[0] + 400 * a = 0
                ------------------------------------------
                x[4] * x[5] * sin(x[2] - 1 / 4) + 
                x[5] * x[6] * sin(x[2] - x[3] - 1 / 4) +
                2 * b * x[5] ** 2 - a * x[1] + 400 * a = 0
                ------------------------------------------
                x[4] * x[6] * sin(x[3] - 1 / 4) + 
                x[5] * x[6] * sin(x[3] - x[2] - 1 / 4) + 
                2 * b * x[6] ** 2 + 881.779 * a = 0
                ------------------------------------------
                a * x[7] + x[4] * x[5] * cos((-1) * x[2] - 1 / 4) + 
                x[4] * x[6] * cos((-1) * x[3] - 1 / 4) - 200 * a - 
                2 * c * x[4] ** 2 + 0.7533e-3 * a * x[4] ** 2 = 0
                ------------------------------------------
                a * x[8] + x[4] * x[5] * cos(x[2] - 1 / 4) + 
                x[5] * x[6] * cos(x[2] - x[3] - 1 / 4) - 2 * c * x[5] ** 2 + 
                0.7533e-3 * a * x[5] ** 2 - 200 * a = 0
                ------------------------------------------
                x[4] * x[6] * cos(x[3] - 1 / 4) + 
                x[5] * x[6] * cos(x[3] - x[2] - 1 / 4) - 
                2 * c * x[6] ** 2 -22.938 * a + 
                0.7533e-3 * a * x[6] ** 2 = 0
            The exact global minimum is approx. known:
                y ~= 5362.069
                x[0] ~= 675.025
                x[1] ~= 1134.021
                x[2] ~= 0.133
                x[3] ~= -0.371
                x[4] ~= 252
                x[5] ~= 252
                x[6] ~= 201.466
                x[7] ~= 426.619
                x[8] ~= 368.488
            Hyperparameters: 
                * The dimension d should be 9
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [0, 0, -0.55, -0.55, 196, 196, 196, -400, -400], 
            [+10, +10, 0.55, 0.55, 252, 252, 252, 800, 800]
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
            'a': 50.176,
            'b': np.sin(0.25),
            'c': np.cos(0.25)
        }

    def _constr_batch(self, X):
        c_1 = -1 * (X[:, 3] - X[:, 2] + .55)
        c_2 = -1 * (X[:, 2] - X[:, 3] + .55)
        c_3 = -1 * (2250000 - X[:, 0] ** 2 - X[:, 7] ** 2)
        c_4 = -1 * (2250000 - X[:, 1] ** 2 - X[:, 8] ** 2)
        c_5 = np.abs(
            X[:, 4] * X[:, 5] * np.sin((-1) * X[:, 2] - 1 / 4) + \
            X[:, 4] * X[:, 6] * np.sin((-1) * X[:, 3] - 1 / 4) + \
            2 * self.parameters['b'] * X[:, 4] ** 2 - self.parameters['a'] * X[:, 0] + \
            400 * self.parameters['a']
        )
        c_6 = np.abs(
            X[:, 4] * X[:, 5] * np.sin(X[:, 2] - 1 / 4) + \
            X[:, 5] * X[:, 6] * np.sin(X[:, 2] - X[:, 3] - 1 / 4) + \
            2 * self.parameters['b'] * X[:, 5] ** 2 - self.parameters['a'] * X[:, 1] + \
            400 * self.parameters['a']
        )
        c_7 = np.abs(
            X[:, 4] * X[:, 6] * np.sin(X[:, 3] - 1 / 4) + \
            X[:, 5] * X[:, 6] * np.sin(X[:, 3] - X[:, 2] - 1 / 4) + \
            2 * self.parameters['b'] * X[:, 6] ** 2 + 881.779 * self.parameters['a']
        )
        c_8 = np.abs(
            self.parameters['a'] * X[:, 7] + X[:, 4] * X[:, 5] * np.cos((-1) * X[:, 2] - 1 / 4) + \
            X[:, 4] * X[:, 6] * np.cos((-1) * X[:, 3] - 1 / 4) - 200 * self.parameters['a'] - \
            2 * self.parameters['c'] * X[:, 4] ** 2 + 0.7533e-3 * self.parameters['a'] * X[:, 4] ** 2
        )
        c_9 = np.abs(
            self.parameters['a'] * X[:, 8] + X[:, 4] * X[:, 5] * np.cos(X[:, 2] - 1 / 4) + \
            X[:, 5] * X[:, 6] * np.cos(X[:, 2] - X[:, 3] - 1 / 4) - 2 * self.parameters['c'] * X[:, 5] ** 2 + \
            0.7533e-3 * self.parameters['a'] * X[:, 5] ** 2 - 200 * self.parameters['a']
        )
        c_10 = np.abs(
            X[:, 4] * X[:, 6] * np.cos(X[:, 3] - 1 / 4) + \
            X[:, 5] * X[:, 6] * np.cos(X[:, 3] - X[:, 2] - 1 / 4) - \
            2 * self.parameters['c'] * X[:, 6] ** 2 -22.938 * self.parameters['a'] + \
            0.7533e-3 * self.parameters['a'] * X[:, 6] ** 2
        )
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 3 * X[:, 0] + 1.0e-6 * X[:, 0] ** 3 + 2 * X[:, 1] + 0.522074e-6 * X[:, 1] ** 3