import numpy as np
from teneva_bm import Bm


class BmHsFunc122(Bm):
    def __init__(self, d=12, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            Problem name: 3 Balls in a Spheric Cage (1988)
            The function 122 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .-------------------------------------.
            | F(x | p, t) -> min s.t. C(x) = True |
            .-------------------------------------.
            p - parameters
                mypi = 4 * atan(1)
                g = 9.80665
                rcage = 1
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
                x[10]
                x[11]
            t - intermediates
                massa = 2700 * (4 / 3) * mypi * x[0] ** 3
                massg = 19300 * (4 / 3) * mypi * x[1] ** 3
                massi = 7860 * (4 / 3) * mypi * x[2] ** 3
            F - objective function
                g * (massa * x[9] + massg * x[10] + massi * x[11])
            C - constraint function
                x[7] + x[1] / 10 = 0
                massa + massg + massi - 0.05 = 0
                (10 - x[0]) - sqrt(x[3] ** 2 + x[6] ** 2 + (x[9] - 10) ** 2) >= 0
                (10 - x[1]) - sqrt(x[4] ** 2 + x[7] ** 2 + (x[10] - 10) ** 2) >= 0
                (10 - x[2]) - sqrt(x[5] ** 2 + x[8] ** 2 + (x[11] - 10) ** 2) >= 0
                sqrt((x[3] - x[4]) ** 2 + (x[6] - x[7]) ** 2 + (x[9] - x[10]) ** 2) - x[0] + x[1] >= 0
                sqrt((x[4] - x[5]) ** 2 + (x[7] - x[8]) ** 2 + (x[10] - x[11]) ** 2) - x[1] + x[2] >= 0
                sqrt((x[5] - x[3]) ** 2 + (x[8] - x[6]) ** 2 + (x[11] - x[9]) ** 2) - x[2] + x[0] >= 0
                (-1) * (x[3] + x[0] / 10) >= 0
                x[4] - x[1] / 10 >= 0
                (-1) * (x[5] + x[2] / 10) >= 0
                (-1) * (x[6] + x[0] / 10) >= 0
                x[8] - x[2] / 10 >= 0
                x[0] - x[1] / 2 >= 0
                x[0] - x[2] / 2 >= 0
                x[1] - x[0] / 2 >= 0
                x[1] - x[2] / 2 >= 0
                x[2] - x[0] / 2 >= 0
                x[2] - x[1] / 2 >= 0
            The exact global minimum is approx. known:
                y ~= 0.004
                x[0] ~= 0.007
                x[1] ~= 0.007
                x[2] ~= 0.007
                x[3] ~= -0.007
                x[4] ~= 0.006
                x[5] ~= -0.009
                x[6] ~= -0.009
                x[7] ~= -0.001
                x[8] ~= 0.005
                x[9] ~= 0.007
                x[10] ~= 0.007
                x[11] ~= 0.007
            Hyperparameters: 
                * The dimension d should be 12
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10], 
            [+10, +10, +10, +10, +10, +10, +10, +10, +10, +10, +10, +10]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 12}

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
            'mypi': 4 * np.arctan(1),
            'g': 9.80665,
            'rcage': 1
        }

    def intermediates(self, X):
        massa = 2700 * (4 / 3) * self.parameters['mypi'] * X[:, 0] ** 3
        massg = 19300 * (4 / 3) * self.parameters['mypi'] * X[:, 1] ** 3
        massi = 7860 * (4 / 3) * self.parameters['mypi'] * X[:, 2] ** 3
        return massa, massg, massi

    def _constr_batch(self, X):
        massa, massg, massi = self.intermediates(X)
        c_1 = np.abs(X[:, 7] + X[:, 1] / 10)
        c_2 = np.abs(massa + massg + massi - 0.05)
        c_3 = -1 * (
            self.parameters['rcage'] - X[:, 0]) - np.sqrt(X[:, 3] ** 2 + 
            X[:, 6] ** 2 + (X[:, 9] - self.parameters['rcage']) ** 2
        )
        c_4 = -1 * (
            self.parameters['rcage'] - X[:, 1]) - np.sqrt(X[:, 4] ** 2 + 
            X[:, 7] ** 2 + (X[:, 10] - self.parameters['rcage']) ** 2
        )
        c_5 = -1 * (
            self.parameters['rcage'] - X[:, 2]) - np.sqrt(X[:, 5] ** 2 + 
            X[:, 8] ** 2 + (X[:, 11] - self.parameters['rcage']) ** 2
        )
        c_6 = -1 * (np.sqrt((X[:, 3] - X[:, 4]) ** 2 + (X[:, 6] - X[:, 7]) ** 2 + (X[:, 9] - X[:, 10]) ** 2) - X[:, 0] + X[:, 1])
        c_7 = -1 * (np.sqrt((X[:, 4] - X[:, 5]) ** 2 + (X[:, 7] - X[:, 8]) ** 2 + (X[:, 10] - X[:, 11]) ** 2) - X[:, 1] + X[:, 2])
        c_8 = -1 * (np.sqrt((X[:, 5] - X[:, 3]) ** 2 + (X[:, 8] - X[:, 6]) ** 2 + (X[:, 11] - X[:, 9]) ** 2) - X[:, 2] + X[:, 0])
        c_9 = -1 * ((-1) * (X[:, 3] + X[:, 0] / 10))
        c_10 = -1 * (X[:, 4] - X[:, 1] / 10)
        c_11 = -1 * ((-1) * (X[:, 5] + X[:, 2] / 10))
        c_12 = -1 * ((-1) * (X[:, 6] + X[:, 0] / 10))
        c_13 = -1 * (X[:, 8] - X[:, 2] / 10)
        c_14 = -1 * (X[:, 0] - X[:, 1] / 2)
        c_15 = -1 * (X[:, 0] - X[:, 2] / 2)
        c_16 = -1 * (X[:, 1] - X[:, 0] / 2)
        c_17 = -1 * (X[:, 1] - X[:, 2] / 2)
        c_18 = -1 * (X[:, 2] - X[:, 0] / 2)
        c_19 = -1 * (X[:, 2] - X[:, 1] / 2)
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, 
                         c_11, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        massa, massg, massi = self.intermediates(X)
        return self.parameters['g'] * (massa * X[:, 9] + massg * X[:, 10] + massi * X[:, 11])