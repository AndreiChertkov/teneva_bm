import numpy as np
from teneva_bm import Bm


class BmHsFunc014(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 014 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0]
                x[1]
            F - objective function
                (x[0] - 2)^2 + (x[1] - 1)^2
            C - equation function
                -0.25 * x[0]^2 - x[1]^2 + 1 >= 0
                x[0] - 2 * x[1] + 1 = 0
            The exact global minimum is known: 
                x[0] = 0.5 * (sqrt(7) - 1) ~ 0.823
                x[1] = 0.25 * (sqrt(7) + 1) ~ 0.911
                y = 9 - 2.875 * sqrt(7) ~ 1.393
            Hyperparameters: 
            * The dimension d should be 2
            * The mode size n may be any (default is 64)
            * The default limits for function inputs are [-10, 10].
        """)
        self.set_grid([-10, -10], [+10, +10])
        self.set_min(
            x=[0.5 * (np.sqrt(7) - 1), 0.25 * (np.sqrt(7) + 1)], 
            y=(9 - 2.875 * np.sqrt(7))
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=False)

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
    
    def _constr_batch(self, X):
        c_1 = -1 * (-0.25 * X[:, 0] ** 2 - X[:, 1] ** 2 + 1)
        c_2 = np.abs(X[:, 0] - 2 * X[:, 1] + 1)
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c
    
    def target_batch(self, X):
        return (X[:, 0] - 2) ** 2 + (X[:, 1] - 1) ** 2