import numpy as np
from teneva_bm import Bm


class BmHsFunc108(Bm):
    def __init__(self, d=9, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 108 from the Hock & Schittkowski collection.
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
                x[8] | >= 0
            F - objective function
                (-1 / 2) * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])
            C - constraint function
                1 - x[2] ** 2 - x[3] ** 2 >= 0
                1 - x[8] ** 2 >= 0
                1 - x[4] ** 2 - x[5] ** 2 >= 0
                1 - x[0] ** 2 - (x[1] - x[8]) ** 2 >= 0
                1 - (x[0] - x[4]) ** 2 - (x[1] - x[5]) ** 2 >= 0
                1 - (x[0] - x[6]) ** 2 - (x[1] - x[7]) ** 2 >= 0
                1 - (x[2] - x[4]) ** 2 - (x[3] - x[5]) ** 2 >= 0
                1 - (x[2] - x[6]) ** 2 - (x[3] - x[7]) ** 2 >= 0
                1 - x[6] ** 2 - (x[7] - x[8]) ** 2 >= 0
                x[0] * x[3] - x[1] * x[2] >= 0
                x[2] * x[8] >= 0
                (-1) * x[4] * x[8] >= 0
                x[4] * x[7] - x[5] * x[6] >= 0
            The exact global minimum is known:
                y = (-1/2) * sqrt(3) 
                aux1 = (2/13) * 13 ** (1/2)
                aux2 = (3/13) * 13 ** (1/2)
                aux3 = (1/13) * 13 ** (1/2) - (3/26) * 13 ** (1/2) * 3 ** (1/2)
                aux4 = (3/26) * 13 ** (1/2) + (1/13) * 13 ** (1/2) * 3 ** (1/2)
                x[0] = aux1 
                x[1] = aux2 
                x[2] = aux3 
                x[3] = aux4 
                x[4] = aux1 
                x[5] = aux2 
                x[6] = aux3 
                x[7] = aux4 
                x[8] = 0
            Hyperparameters: 
                * The dimension d should be 9
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10, -10, -10, -10, -10, -10, 0], [+10, +10, +10, +10, +10, +10, +10, +10, +10])
        aux1 = (2/13) * 13 ** (1/2)
        aux2 = (3/13) * 13 ** (1/2)
        aux3 = (1/13) * 13 ** (1/2) - (3/26) * 13 ** (1/2) * 3 ** (1/2)
        aux4 = (3/26) * 13 ** (1/2) + (1/13) * 13 ** (1/2) * 3 ** (1/2)
        self.set_min(x=[aux1, aux2, aux3, aux4, aux1, aux2, aux3, aux4, 0], y=(-1/2) * np.sqrt(3))
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

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

    def _constr_batch(self, X):
        c_1 = -1 * (1 - X[:, 2] ** 2 - X[:, 3] ** 2)
        c_2 = -1 * (1 - X[:, 8] ** 2)
        c_3 = -1 * (1 - X[:, 4] ** 2 - X[:, 5] ** 2)
        c_4 = -1 * (1 - X[:, 0] ** 2 - (X[:, 1] - X[:, 8]) ** 2)
        c_5 = -1 * (1 - (X[:, 0] - X[:, 4]) ** 2 - (X[:, 1] - X[:, 5]) ** 2)
        c_6 = -1 * (1 - (X[:, 0] - X[:, 6]) ** 2 - (X[:, 1] - X[:, 7]) ** 2)
        c_7 = -1 * (1 - (X[:, 2] - X[:, 4]) ** 2 - (X[:, 3] - X[:, 5]) ** 2)
        c_8 = -1 * (1 - (X[:, 2] - X[:, 6]) ** 2 - (X[:, 3] - X[:, 7]) ** 2)
        c_9 = -1 * (1 - X[:, 6] ** 2 - (X[:, 7] - X[:, 8]) ** 2)
        c_10 = -1 * (X[:, 0] * X[:, 3] - X[:, 1] * X[:, 2])
        c_11 = -1 * (X[:, 2] * X[:, 8])
        c_12 = -1 * ((-1) * X[:, 4] * X[:, 8])
        c_13 = -1 * (X[:, 4] * X[:, 7] - X[:, 5] * X[:, 6])
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (-1 / 2) * (X[:, 0] * X[:, 3] - X[:, 1] * X[:, 2] + X[:, 2] * X[:, 8] - \
                X[:, 4] * X[:, 8] + X[:, 4] * X[:, 7] - X[:, 5] * X[:, 6])