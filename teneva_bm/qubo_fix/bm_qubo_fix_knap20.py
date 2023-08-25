import numpy as np
from teneva_bm import Bm


class BmQuboFixKnap20(Bm):
    def __init__(self, d=20, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Binary 20-dimensional knapsack problem
            -1 * sum p_i x_i -> min s.t. sum w_i x_i < C
            with fixed weights w, profits p and the capacity C.
            The exact minimum is known (-1024). We use the parameters from
            the work Dong et al. 2021 (problem k2), where phase angle
            modulated bat algorithm (P-AMBA) was proposed for binary problems
            with application to antenna topology optimization.
        """)

        i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
        self.set_min(i=i, y=-1024)

        self.set_constr(penalty=0.)

    @property
    def args_constr(self):
        return {'d': 20, 'n': 2}

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.zeros(20, dtype=int)
        for k in [0, 5, 8, 12]:
            i[k] = 1
        return np.array(i, dtype=int), -169.0

    @property
    def with_constr(self):
        return True

    def constr(self, i):
        return np.dot(self._w, i) - self._C

    def prep_bm(self):
        self._w = np.array([
            92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70,
            48, 14, 58])

        self._p = np.array([
            44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17, 75,
            29, 75, 63])

        self._C = 878

    def target(self, i):
        return -np.dot(self._p, i)
