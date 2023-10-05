import numpy as np
from teneva_bm import Bm


class BmHsFunc054(Bm):
    def __init__(self, d=6, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 054 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | t) -> min s.t. C(x) = True |
            .----------------------------------.
            x - continuous control
                x[0] | >= 0   | <= 20000
                x[1] | >= -10 | <= 10
                x[2] | >= 0   | <= 10000000
                x[3] | >= 0   | <= 20
                x[4] | >= -1  | <= 1
                x[5] | >= 0   | <= 200000000
            t - intermediates
                y[0] = (x[0] - 10000) / 8000
                y[1] = (x[1] - 1) / 1
                y[2] = (x[2] - 2000000) / 7000000
                y[3] = (x[3] - 10) / 50
                y[4] = (x[4] - 1 / 1000) * 20
                y[5] = (x[5] - 100000000) / 500000000
                h1 = (y[0] ** 2 + y[0] * y[1] * 2 / 5 + y[1] ** 2) * 25 / 24
                h2 = y[2] ** 2 + y[3] ** 2 + y[4] ** 2 + y[5] ** 2
            F - objective function
                -exp(-(h1 + h2) / 2)
            C - constraint function
                x[0] + 4000 * x[1] - 17600 = 0
            The exact global minimum is known:
                y = -exp(-27/280) 
                x[0] = 91600/7 
                x[1] = 79/70 
                x[2] = 2000000
                x[3] = 10
                x[4] = 1/1000 
                x[5] = 100000000
            Hyperparameters: 
                * The dimension d should be 6
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, -10, 0, 0, -1, 0], [20000, 10, 10000000, 20, 1, 200000000])
        self.set_min(x=[91600/7, 79/70, 2000000, 10, 1/1000, 100000000], y=-np.exp(-27/280))
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

    def intermediates(self, X):        
        y = np.zeros_like(X)
        y[:, 0] = (X[:, 0] - 10000) / 8000
        y[:, 1] = (X[:, 1] - 1) / 1
        y[:, 2] = (X[:, 2] - 2000000) / 7000000
        y[:, 3] = (X[:, 3] - 10) / 50
        y[:, 4] = (X[:, 4] - 1 / 1000) * 20
        y[:, 5] = (X[:, 5] - 100000000) / 500000000
        h1 = (y[:, 0] ** 2 + y[:, 0] * y[:, 1] * 2 / 5 + y[:, 1] ** 2) * 25 / 24
        h2 = y[:, 2] ** 2 + y[:, 3] ** 2 + y[:, 4] ** 2 + y[:, 5] ** 2
        return h1, h2

    def constr_batch(self, X):
        return np.abs(X[:, 0] + 4000 * X[:, 1] - 17600)

    def target_batch(self, X):
        h1, h2 = self.intermediates(X)
        return -np.exp(-(h1 + h2) / 2)