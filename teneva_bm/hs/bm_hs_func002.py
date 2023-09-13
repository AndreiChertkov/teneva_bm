import numpy as np
from teneva_bm import Bm


class BmHsFunc002(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 002 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .-------------.
            | F(x) -> min |
            .-------------.
            x - continuous control
                x[0] 
                x[1] | >= 3/2
            F - objective function
                100 * (x[1] - x[0]^2)^2 + (1 - x[0])^2
            The exact global minimum is known: x = [~1.224, 1.5], y ~ 0.05.
            Hyperparameters: 
            * The dimension d should be 2
            * The mode size n may be any (default is 64)
            * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10., 3/2], +10.)
        self.set_min(x=[1.224, 1.5], y=0.05042618789360708)

    @property
    def args_constr(self):
        return {'d': 2}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [0] * self.d
        return np.array(i, dtype=int), -1

    def target_batch(self, X):
        return 100. * (X[:, 1] - X[:, 0] ** 2) ** 2 + (1. - X[:, 0]) ** 2