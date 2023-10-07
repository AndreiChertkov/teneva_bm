import numpy as np
from teneva_bm import Bm


class BmHsFunc067(Bm):
    def __init__(self, d=5, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 067 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | t) -> min s.t. C(x) = True |
            .----------------------------------.
            x - continuous control
                x[0] | >= 1e-5 | <= 2000
                x[1] | >= 1e-5 | <= 16000
                x[2] | >= 1e-5 | <= 120
                x[3]
                x[4]
            t - intermediates
                t_1 = x[3]
                t_2 = 1.22 * t_1 - x[0]
                t_3 = (x[1] + t_2) / x[0]
                t_4 = 0.01 * x[0] * (112 + 13.167 * t_3 - 0.6667 * t_3 ** 2)
                t_5 = x[4]
                t_6 = 86.35 + 1.098 * t_3 - 0.038 * t_3 ** 2 + 0.325 * (t_5 - 89)
                t_7 = 3 * t_6 - 133
                t_8 = 35.82 - 0.222 * y8
                t_9 = 98000 * x[2] / (t_1 * t_8 + 1000 * x[2])                      
            F - objective function
                -(0.063 * t_1 * t_6 - 5.04 * x[0] - 3.36 * t_2 - 0.035 * x[1] - 10 * x[2])
            C - constraint function
                t_4 - t_1 = 0
                t_9 - t_5 = 0
                t_1 - 0 >= 0
                t_2 - 0 >= 0
                t_5 - 85 >= 0
                t_6 - 90 >= 0
                t_3 - 3 >= 0
                t_8 - 1 / 100 >= 0
                t_7 - 145 >= 0
                5000 - t_1 >= 0
                2000 - t_2 >= 0
                93 - t_5 >= 0 
                95 - t_6 >= 0
                12 - t_3 >= 0
                4 - t_8 >= 0
                162 - t_7 >= 0
            The exact global minimum is approx. known:
                y ~= -1162.027
                x[0] ~= 1728.371
                x[1] ~= 16000
                x[2] ~= 98.136
                x[3] ~= 3056.042
                x[4] ~= 90.619
            Hyperparameters: 
                * The dimension d should be 5
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([1e-5, 1e-5, 1e-5, -10, -10], [2000, 16000, 120, +10, +10])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 5}

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
        t_1 = X[:, 3]
        t_2 = 1.22 * t_1 - X[:, 0]
        t_3 = (X[:, 1] + t_2) / X[:, 0]
        t_4 = 0.01 * X[:, 0] * (112 + 13.167 * t_3 - 0.6667 * t_3 ** 2)
        t_5 = X[:, 4]
        t_6 = 86.35 + 1.098 * t_3 - 0.038 * t_3 ** 2 + 0.325 * (t_5 - 89)
        t_7 = 3 * t_6 - 133
        t_8 = 35.82 - 0.222 * t_7
        t_9 = 98000 * X[:, 2] / (t_1 * t_8 + 1000 * X[:, 2])  
        return t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9
    
    def _constr_batch(self, X):
        t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9 = self.intermediates(X)
        c_1 = np.abs(t_4 - t_1)
        c_2 = np.abs(t_9 - t_5)
        c_3 = -1 * (t_1 - 0)
        c_4 = -1 * (t_2 - 0)
        c_5 = -1 * (t_5 - 85)
        c_6 = -1 * (t_6 - 90)
        c_7 = -1 * (t_3 - 3)
        c_8 = -1 * (t_8 - 1 / 100)
        c_9 = -1 * (t_7 - 145)
        c_10 = -1 * (5000 - t_1)
        c_11 = -1 * (2000 - t_2)
        c_12 = -1 * (93 - t_5)
        c_13 = -1 * (95 - t_6)
        c_14 = -1 * (12 - t_3)
        c_15 = -1 * (4 - t_8)
        c_16 = -1 * (162 - t_7)
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, 
                         c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        t = self.intermediates(X)
        return -(0.063 * t[0] * t[5] - 5.04 * X[:, 0] - 3.36 * t[1] - 0.035 * X[:, 1] - 10 * X[:, 2])