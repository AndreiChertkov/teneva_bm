import numpy as np
from teneva_bm import Bm


class BmHsFunc008(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 008 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0]
                x[1]
            F - objective function
                -1
            C - equation function
                x[0]^2 + x[1]^2 - 25 = 0
                x[0] * x[1] - 9 = 0
            The exact global minimum is known: 
                x[0] = sqrt((25 + sqrt(301))/2) = ~4.602
                x[1] = sqrt((25 - sqrt(301))/2) = ~1.956
                y = -1
            Hyperparameters: 
            * The dimension d should be 2
            * The mode size n may be any (default is 64)
            * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(-10., +10.)
        self.set_min(
            x=[np.sqrt((25 + np.sqrt(301))/2), np.sqrt((25 - np.sqrt(301))/2)], 
            y=-1
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

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
    def with_constr(self):
        return True
        
    def constr_batch(self, X):
        c_1 = np.abs(X[:, 0] ** 2 + X[:, 1] ** 2 - 25) 
        c_2 = np.abs(X[:, 0] * X[:, 1] - 9)
        return c_1 + c_2

    def target_batch(self, X):
        return np.ones(X.shape[0]) * -1
