import numpy as np
from teneva_bm import Bm


class BmHsFunc045(Bm):
    def __init__(self, d=5, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 045 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .-------------.
            | F(x) -> min |
            .-------------.
            x - continuous control
                x[0] | >= 0 | <= 1
                x[1] | >= 0 | <= 2
                x[2] | >= 0 | <= 3
                x[3] | >= 0 | <= 4
                x[4] | >= 0 | <= 5
            F - objective function
                2 - (1 / 120) * x[0] * x[1] * x[2] * x[3] * x[4]
            The exact global minimum is known:
                y = 1
                x[0] = 1
                x[1] = 2
                x[2] = 3
                x[3] = 4
                x[4] = 5
            Hyperparameters: 
                * The dimension d should be 5
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0, 0, 0], [1, 2, 3, 4, 5])
        self.set_min(x=[1, 2, 3, 4, 5], y=1)

    @property
    def args_constr(self):
        return {'d': 5}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    def target_batch(self, X):
        return 2 - (1 / 120) * X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3] * X[:, 4]