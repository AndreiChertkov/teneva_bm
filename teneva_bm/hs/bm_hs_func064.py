import numpy as np
from teneva_bm import Bm


class BmHsFunc064(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 064 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 1e-5
                x[1] | >= 1e-5
                x[2] | >= 1e-5
            F - objective function
                5 * x[0] + 50000 / x[0] + 20 * x[1] + 72000 / x[1] + 10 * x[2] + 144000 / x[2]
            C - constraint function
                1 - 4 / x[0] - 32 / x[1] - 120 / x[2] >= 0
            The exact global minimum is approx. known:
                y ~= 6299.842
                x[0] ~= 108.735
                x[1] ~= 85.126
                x[2] ~= 204.325
            Hyperparameters: 
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([1e-5, 1e-5, 1e-5], [1e+5, 1e+5, 1e+5]) # extended right boundaries
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
        return -1 * (1 - 4 / X[:, 0] - 32 / X[:, 1] - 120 / X[:, 2])

    def target_batch(self, X):
        return 5 * X[:, 0] + 50000 / X[:, 0] + 20 * X[:, 1] + 72000 / X[:, 1] + 10 * X[:, 2] + 144000 / X[:, 2]