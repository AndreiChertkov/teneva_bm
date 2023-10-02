import numpy as np
from teneva_bm import Bm


class BmHsFunc042(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 042 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0]
                x[1]
                x[2]
                x[3]
            F - objective function
                (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 3) ** 2 + (x[3] - 4) ** 2
            C - constraint function
                x[0] - 2 = 0
                x[2] ** 2 + x[3] ** 2 - 2 = 0
            The exact global minimum is known:
                y = 28 - 10 * sqrt(2) 
                x[0] = 2
                x[1] = 2
                x[2] = (3/5) * sqrt(2) 
                x[3] = (4/5) * sqrt(2) 
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10, -10], [+10, +10, +10, +10])
        self.set_min(x=[2, 2, (3/5) * np.sqrt(2), (4/5) * np.sqrt(2)], y=28 - 10 * np.sqrt(2))
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

    def _constr_batch(self, X):
        c_1 = np.abs(X[:, 0] - 2)
        c_2 = np.abs(X[:, 2] ** 2 + X[:, 3] ** 2 - 2)
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (X[:, 0] - 1) ** 2 + (X[:, 1] - 2) ** 2 + (X[:, 2] - 3) ** 2 + (X[:, 3] - 4) ** 2