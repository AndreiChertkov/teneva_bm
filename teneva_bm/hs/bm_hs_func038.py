import numpy as np
from teneva_bm import Bm


class BmHsFunc038(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 038 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .-------------.
            | F(x) -> min |
            .-------------.
            x - continuous control
                x[0] | >= -10 | <= 10
                x[1] | >= -10 | <= 10
                x[2] | >= -10 | <= 10
                x[3] | >= -10 | <= 10
            F - objective function
                100 * (x[1] - x[0] ** 2) ** 2 + 
                (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2 + 
                (1 - x[2]) ** 2 + 10.1 * ((x[1] - 1) ** 2 + 
                (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)
            The exact global minimum is known:
                y = 0
                x[0] = 1
                x[1] = 1
                x[2] = 1
                x[3] = 1
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10, -10], [10, 10, 10, 10])
        self.set_min(x=[1, 1, 1, 1], y=0)

    @property
    def args_constr(self):
        return {'d': 4}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    def target_batch(self, X):
        return 100 * (X[:, 1] - X[:, 0] ** 2) ** 2 + (1 - X[:, 0]) ** 2 + \
                90 * (X[:, 3] - X[:, 2] ** 2) ** 2 + (1 - X[:, 2]) ** 2 + \
                10.1 * ((X[:, 1] - 1) ** 2 + (X[:, 3] - 1) ** 2) + 19.8 * (X[:, 1] - 1) * (X[:, 3] - 1)