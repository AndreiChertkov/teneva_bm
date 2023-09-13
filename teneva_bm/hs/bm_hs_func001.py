import numpy as np
from teneva_bm import Bm


class BmHsFunc001(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 001 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .-------------.
            | F(x) -> min |
            .-------------.
            x - continuous control
                x[0]
                x[1] | >= -3/2
            F - objective function
                100 * (x[1] - x[0]^2)^2 + (1 - x[0])^2
            The exact global minimum is known: x = [1, 1], y = 0.
            Hyperparameters: 
            * The dimension d should be 2
            * The mode size n may be any (default is 64)
            * The default limits for function inputs are [-10, 10].
        """)
        
        self.set_grid([-10., -3/2], +10.)
        self.set_min(x=1., y=0.)

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
        i = [21, 42]
        return np.array(i, dtype=int), 90015.99999999996

    def target_batch(self, X):
        return 100. * (X[:, 1] - X[:, 0] ** 2) ** 2 + (1. - X[:, 0]) ** 2

# self.set_desc("""
#     DRAFT!!! The function 001 from the Hock & Schittkowski collection.
#     The dimension should be 2, and the mode size may be any (default is
#     64), the default limits for function inputs are [-10, 10].
#     The exact global minimum is known: x = [1, 1], y = 0.
# """)