import numpy as np
from teneva_bm import Bm


class BmHsFunc087(Bm):
    def __init__(self, d=15, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 087 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a = 131.078
                b = 1.48477
                c = 0.90798
                d = cos(1.47588)
                e = sin(1.47588)
            x - continuous control
                x[0] | >= 0     | <= 400
                x[1] | >= 0     | <= 1000
                x[2] | >= 340   | <= 420
                x[3] | >= 340   | <= 420
                x[4] | >= -1000 | <= 1000
                x[5] | >= 0     | <= 0.5236
                x[6] | >= 0     | <= 300
                x[7] | >= 0     | <= 300
                x[8] | >= 0     | <= 900
                x[9] | >= 0     | <= 900
                x[10] | >= 0    | <= 800
                x[11] | >= 0    | <= 800
                x[12] | >= -1   | <= 1
                x[13] | >= -1   | <= 1
                x[14] | >= -1   | <= 1
            F - objective function
                30 * x[0] * (1 - x[12]) / 2 + 31 * x[0] * (1 + x[12]) / 2 + \
                28 * x[1] * (1 - x[13]) / 2 + 29 * x[1] * (1 + x[13]) * (1 - x[14]) / 4 + \
                30 * x[1] * (1 + x[14]) / 2 x[8] * x[9] + x[6] * x[7] + x[10] * x[11]
            C - constraint function
                300 - x[0] - x[2] * x[3] * cos(b - x[5]) / a + c * d * x[2] ** 2 / a >= 0
                (-1) * x[1] - x[2] * x[3] * cos(b + x[5]) / a + c * d * x[3] ** 2 / a >= 0
                (-1) * x[4] - x[2] * x[3] * sin(b + x[5]) / a + c * e * x[3] ** 2 / a >= 0
                200 - x[2] * x[3] * sin(b - x[5]) / a + c * e * x[2] ** 2 / a >= 0
                x[0] - 300 + x[6] - x[7] >= 0
                x[1] - 100 + x[8] - x[9] >= 0
                x[1] - 200 + x[10] - x[11] >= 0
                x[12] * (x[6] + x[7]) - x[0] - 300 >= 0
                x[13] * (x[8] + x[9]) - x[1] - 100 >= 0
                x[14] * (x[10] + x[11]) - x[1] - 200 >= 0
            The exact global minimum is approx. known:
                y ~= 8853.540
                x[0] ~= 201.785
                x[1] ~= 100
                x[2] ~= 383.071
                x[3] ~= 420
                x[4] ~= -10.908
                x[5] ~= 0.073
                x[6] ~= 98.215
                x[7] ~= 0
                x[8] ~= 0
                x[9] ~= 0
                x[10] ~= 100
                x[11] ~= 0
                x[12] ~= -1
                x[13] ~= -1
                x[14] ~= -1
            Hyperparameters: 
                * The dimension d should be 15
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [0, 0, 340, 340, -1000, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1], 
            [400, 1000, 420, 420, 1000, 0.5236, 300, 300, 900, 900, 800, 800, 1, 1, 1]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 15}

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
            'a': 131.078,
            'b': 1.48477,
            'c': 0.90798,
            'd': np.cos(1.47588),
            'e': np.sin(1.47588)
        }

    def _constr_batch(self, X):
        c_1 = -1 * (
            300 - X[:, 0] - X[:, 2] * X[:, 3] * np.cos(self.parameters['b'] - X[:, 5]) / self.parameters['a'] + \
            self.parameters['c'] * self.parameters['d'] * X[:, 2] ** 2 / self.parameters['a']
        )
        c_2 = -1 * (
            (-1) * X[:, 1] - X[:, 2] * X[:, 3] * np.cos(self.parameters['b'] + X[:, 5]) / self.parameters['a'] + \
            self.parameters['c'] * self.parameters['d'] * X[:, 3] ** 2 / self.parameters['a']
        )
        c_3 = -1 * (
            (-1) * X[:, 4] - X[:, 2] * X[:, 3] * np.sin(self.parameters['b'] + X[:, 5]) / self.parameters['a'] + \
            self.parameters['c'] * self.parameters['e'] * X[:, 3] ** 2 / self.parameters['a']
        )
        c_4 = -1 * (
            200 - X[:, 2] * X[:, 3] * np.sin(self.parameters['b'] - X[:, 5]) / self.parameters['a'] + \
            self.parameters['c'] * self.parameters['e'] * X[:, 2] ** 2 / self.parameters['a']
        )
        c_5 = -1 * (X[:, 0] - 300 + X[:, 6] - X[:, 7])
        c_6 = -1 * (X[:, 1] - 100 + X[:, 8] - X[:, 9])
        c_7 = -1 * (X[:, 1] - 200 + X[:, 10] - X[:, 11])
        c_8 = -1 * (X[:, 12] * (X[:, 6] + X[:, 7]) - X[:, 0] - 300)
        c_9 = -1 * (X[:, 13] * (X[:, 8] + X[:, 9]) - X[:, 1] - 100)
        c_10 = -1 * (X[:, 14] * (X[:, 10] + X[:, 11]) - X[:, 1] - 200)
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 30 * X[:, 0] * (1 - X[:, 12]) / 2 + 31 * X[:, 0] * (1 + X[:, 12]) / 2 + \
               28 * X[:, 1] * (1 - X[:, 13]) / 2 + 29 * X[:, 1] * (1 + X[:, 13]) * (1 - X[:, 14]) / 4 + \
               30 * X[:, 1] * (1 + X[:, 14]) / 2 + X[:, 8] * X[:, 9] + X[:, 6] * X[:, 7] + X[:, 10] * X[:, 11]