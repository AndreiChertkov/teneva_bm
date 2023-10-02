import numpy as np
from teneva_bm import Bm


class BmHsFunc041(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 041 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0 | <= 1
                x[1] | >= 0 | <= 1
                x[2] | >= 0 | <= 1
                x[3] | >= 0 | <= 2
            F - objective function
                2 - x[0] * x[1] * x[2]
            C - constraint function
                x[0] + 2 * x[1] + 2 * x[2] - x[3] = 0
            The exact global minimum is known:
                y = 52/27 
                x[0] = 2/3 
                x[1] = 1/3 
                x[2] = 1/3 
                x[3] = 2
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0, 0], [1, 1, 1, 2])
        self.set_min(x=[2/3, 1/3, 1/3, 2], y=52/27)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 4}

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
        return np.abs(X[:, 0] + 2 * X[:, 1] + 2 * X[:, 2] - X[:, 3])

    def target_batch(self, X):
        return 2 - X[:, 0] * X[:, 1] * X[:, 2]