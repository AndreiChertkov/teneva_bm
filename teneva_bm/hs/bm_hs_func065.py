import numpy as np
from teneva_bm import Bm


class BmHsFunc065(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 065 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= -4.5 | <= 4.5
                x[1] | >= -4.5 | <= 4.5
                x[2] | >= -5 | <= 5
            F - objective function
                (x[0] - x[1]) ** 2 + (x[0] + x[1] - 10) ** 2 / 9 + (x[2] - 5) ** 2
            C - constraint function
                48 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 >= 0
            The exact global minimum is approx. known:
                y ~= 0.954
                x[0] ~= 3.650
                x[1] ~= 3.650
                x[2] ~= 4.620
            Hyperparameters: 
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-4.5, -4.5, -5], [4.5, 4.5, 5])
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
        return -1 * (48 - X[:, 0] ** 2 - X[:, 1] ** 2 - X[:, 2] ** 2)

    def target_batch(self, X):
        return (X[:, 0] - X[:, 1]) ** 2 + (X[:, 0] + X[:, 1] - 10) ** 2 / 9 + (X[:, 2] - 5) ** 2