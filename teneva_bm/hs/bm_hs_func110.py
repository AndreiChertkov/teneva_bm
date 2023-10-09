import numpy as np
from teneva_bm import Bm


class BmHsFunc110(Bm):
    def __init__(self, d=10, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 110 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .-------------.
            | F(x) -> min |
            .-------------.
            x - continuous control
                x[0] | >= 2.001 | <= 9.999
                x[1] | >= 2.001 | <= 9.999
                x[2] | >= 2.001 | <= 9.999
                x[3] | >= 2.001 | <= 9.999
                x[4] | >= 2.001 | <= 9.999
                x[5] | >= 2.001 | <= 9.999
                x[6] | >= 2.001 | <= 9.999
                x[7] | >= 2.001 | <= 9.999
                x[8] | >= 2.001 | <= 9.999
                x[9] | >= 2.001 | <= 9.999
            F - objective function
                sum(log(x - 2) ** 2 + log(10 - x) ** 2) - prod(x) ** (1 / 5)
            The exact global minimum is approx. known:
                y ~= -45.778
                x[0] ~= 9.350
                x[1] ~= 9.350
                x[2] ~= 9.350
                x[3] ~= 9.350
                x[4] ~= 9.350
                x[5] ~= 9.350
                x[6] ~= 9.350
                x[7] ~= 9.350
                x[8] ~= 9.350
                x[9] ~= 9.350
            Hyperparameters: 
                * The dimension d should be 10
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001, 2.001], 
            [9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999, 9.999]
        )

    @property
    def args_constr(self):
        return {'d': 10}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    def target_batch(self, X):
        return (np.log(X - 2) ** 2 + np.log(10 - X) ** 2).sum(-1) - np.prod(X) ** (1 / 5)