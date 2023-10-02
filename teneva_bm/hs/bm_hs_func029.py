import numpy as np
from teneva_bm import Bm


class BmHsFunc029(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 029 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] 
                x[1]
                x[2]
            F - objective function
                -x[0] * x[1] * x[2]
            C - constraint function
                (-1) * x[0] ** 2 - 2 * x[1] ** 2 - 4 * x[2] ** 2 + 48 >= 0
            The exact global minimum is known:
                y = -16 * sqrt(2)
                x[0] = 4
                x[1] = 2 * sqrt(2)
                x[2] = 2
            Hyperparameters: 
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10], [+10, +10, +10])
        self.set_min(x=[4, 2 * np.sqrt(2), 2], y=-16 * np.sqrt(2))
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
        return -1 * ((-1) * X[:, 0] ** 2 - 2 * X[:, 1] ** 2 - 4 * X[:, 2] ** 2 + 48)

    def target_batch(self, X):
        return -X[:, 0] * X[:, 1] * X[:, 2]