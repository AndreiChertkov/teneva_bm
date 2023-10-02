import numpy as np
from teneva_bm import Bm


class BmHsFunc035(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 035 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0
                x[1] | >= 0
                x[2] | >= 0
            F - objective function
                9 - 8 * x[0] - 6 * x[1] - 4 * x[2] + 
                2 * x[0] ** 2 + 2 * x[1] ** 2 + 
                x[2] ** 2 + 2 * x[0] * x[1] + 2 * x[0] * x[2]
            C - constraint function
                3 - x[0] - x[1] - 2 * x[2] >= 0
            The exact global minimum is known:
                y = 1/9 
                x[0] = 4/3 
                x[1] = 7/9 
                x[2] = 4/9 
            Hyperparameters: 
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0], [+10, +10, +10])
        self.set_min(x=[4/3, 7/9, 4/9], y=1/9)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 3}

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
        return -1 * (3 - X[:, 0] - X[:, 1] - 2 * X[:, 2])

    def target_batch(self, X):
        return 9 - 8 * X[:, 0] - 6 * X[:, 1] - 4 * X[:, 2] + \
               2 * X[:, 0] ** 2 + 2 * X[:, 1] ** 2 + X[:, 2] ** 2 + \
               2 * X[:, 0] * X[:, 1] + 2 * X[:, 0] * X[:, 2]