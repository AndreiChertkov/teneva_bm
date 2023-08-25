import numpy as np
from teneva_bm import Bm


class BmHsFunc006(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            DRAFT!!! The function 006 from the Hock & Schittkowski collection
            with the explicit constraint. The dimension should be 2, and the
            mode size may be any (default is 64), the default limits for
            function inputs are [-10, 10]. The exact global minimum is known:
            x = [1, 1], y = 0. Note that the default penalty for the constraint
            is "1.E+3" and the amplitude of the constraint is used.
        """)

        self.set_grid(-10., +10.)
        # TODO: do we need the shift as in "func" collection???

        self.set_min(x=1., y=0.)

        # TODO: is it ok, to set such constraint by default???
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
    def ref(self):
        i = [10, 12]
        return np.array(i, dtype=int), 688649.2545012803

    @property
    def with_constr(self):
        return True

    def constr_batch(self, X):
        return np.abs(10. * (X[:, 1] - X[:, 0]**2))

    def target_batch(self, X):
        return (1. - X[:, 0])**2
