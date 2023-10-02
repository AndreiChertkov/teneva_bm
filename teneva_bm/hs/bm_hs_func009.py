import numpy as np
from teneva_bm import Bm


class BmHsFunc009(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 009 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0]
                x[1]
            F - objective function
                sin(4 * atan(1) * x[0] / 12) * cos(4 * atan(1) * x[1] / 16)
            C - equation function
                4 * x[0] - 3 * x[1] = 0
            The exact global minimum is known: 
                y = -0.5
                x[0] = -3
                x[1] = -4
            Hyperparameters: 
            * The dimension d should be 2
            * The mode size n may be any (default is 64)
            * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10], [+10, +10])
        self.set_min(x=[-3, -4], y=-0.5)
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
        return np.abs(4 * X[:, 0] - 3 * X[:, 1])

    def target_batch(self, X):
        c = np.arctan(1) / 12
        return np.sin(4 * c * X[:, 0]) * np.cos(3 * c * X[:, 1])
