import numpy as np
from teneva_bm import Bm


class BmHsFunc062(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 062 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0 | <= 1
                x[1] | >= 0 | <= 1
                x[2] | >= 0 | <= 1
            F - objective function
                (-32.174) * (255 * log((x[0] + x[1] + x[2] + 0.03) / (0.09 * x[0] + x[1] + x[2] + 0.03)) + 
                280 * log((x[1] + x[2] + 0.03) / (0.07 * x[1] + x[2] + 0.03)) + 
                290 * log((x[2] + 0.03) / (0.13 * x[2] + 0.03)))
            C - constraint function
                x[0] + x[1] + x[2] - 1 = 0
            The exact global minimum is approx. known:
                y ~= -26272.514
                x[0] ~= 0.618
                x[1] ~= 0.328
                x[2] ~= 0.054
            Hyperparameters: 
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0], [1, 1, 1])
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
        return np.abs(X[:, 0] + X[:, 1] + X[:, 2] - 1)

    def target_batch(self, X):
        return (-32.174) * (255 * np.log(
            (X[:, 0] + X[:, 1] + X[:, 2] + 0.03) / 
            (0.09 * X[:, 0] + X[:, 1] + X[:, 2] + 0.03)
        ) + 280 * np.log(
            (X[:, 1] + X[:, 2] + 0.03) / 
            (0.07 * X[:, 1] + X[:, 2] + 0.03)
        ) + 290 * np.log(
            (X[:, 2] + 0.03) / 
            (0.13 * X[:, 2] + 0.03)
        ))