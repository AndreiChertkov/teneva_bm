import numpy as np
from teneva_bm import Bm


class BmHsFunc001(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            DRAFT!!! The function 001 from the Hock & Schittkowski collection.
            The dimension should be 2, and the mode size may be any (default is
            64), the default limits for function inputs are [-10, 10].
            The exact global minimum is known: x = [1, 1], y = 0.
        """)

        self.set_grid(-10., +10.)
        # TODO: do we need the shift as in "func" collection???

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
        return 100. * (X[:, 1] - X[:, 0]**2)**2 + (1. - X[:, 0])**2
