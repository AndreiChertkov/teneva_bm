import numpy as np
from teneva_bm import Bm


class BmHsFunc054_3(Bm):
    def __init__(self, d=6, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 054 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | t) -> min s.t. C(x) = True |
            .----------------------------------.
            x - continuous control
                x[0] | >= -5/4     | <= 5/4
                x[1] | >= -11      | <= 9
                x[2] | >= -2/7     | <= 8/7
                x[3] | >= -1/5     | <= 1/5
                x[4] | >= -1001/50 | <= 999/50
                x[5] | >= -1/5     | <= 1/5
            t - intermediates
                h1 = (x[0] ** 2 + x[0] * x[1] * 2 / 5 + x[1] ** 2) * 25 / 24
                h2 = x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2
                mf = h1 + h2
            F - objective function
                mf
            C - constraint function
                8 * x[0] + 4 * x[1] - 18 / 5 = 0
            The exact global minimum is known:
                y = 27/140 
                x[0] = 27/70 
                x[1] = 9/70 
                x[2] = 0
                x[3] = 0
                x[4] = 0
                x[5] = 0
            Hyperparameters: 
                * The dimension d should be 6
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-5/4, -11, -2/7, -1/5, -1001/50, -1/5], [5/4, 9, 8/7, 1/5, 999/50, 1/5])
        self.set_min(x=[27/70, 9/70, 0, 0, 0, 0], y=27/140)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 6}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def with_constr(self):
        return True

    def constr_batch(self, X):
        return np.abs(8 * X[:, 0] + 4 * X[:, 1] - 18 / 5)

    def target_batch(self, X):
        h1 = (X[:, 0] ** 2 + X[:, 0] * X[:, 1] * 2 / 5 + X[:, 1] ** 2) * 25 / 24
        h2 = X[:, 2] ** 2 + X[:, 3] ** 2 + X[:, 4] ** 2 + X[:, 5] ** 2
        obj = h1 + h2
        return obj