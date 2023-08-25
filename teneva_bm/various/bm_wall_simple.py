import numpy as np
from teneva_bm import Bm


class BmWallSimple(Bm):
    def __init__(self, d=10, n=50, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Simple example of the special discrete function ("wall"), which is
            difficult to optimize by tensor methods. The exact global minimum
            is known: i = [0, ..., 0], y = 0. The target function returns "0" if
            the requested multi-index is optimal; returns a large number ("10 *
            d") if the requested multi-index matches the optimal one in at
            least one element; and returns the first element of the multi-index
            otherwise. The dimension and mode size may be any (default are
            d=10, n=50).
        """)

        self.set_min(i=0, y=0.)

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.ones(10, dtype=int)
        for k in [1, 3, 6]:
            i[k] = 10
        for k in [0, 2, 4, 9]:
            i[k] = 5
        return np.array(i, dtype=int), 5.

    def target(self, i):
        if len(np.where(i == self.i_min_real)[0]) == self.d:
            return 0.
        elif len(np.where(i == self.i_min_real)[0]) > 0:
            return self.d * 10.
        else:
            return i[0] * 1.
