import numpy as np
from teneva_bm import Bm


class BmQuboFixKnap50(Bm):
    def __init__(self, d=50, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Binary 50-dimensional knapsack problem
            -1 * sum p_i x_i -> min s.t. sum w_i x_i < C
            with fixed weights w, profits p and the capacity C.
            The exact minimum is known (-3103). We use the parameters from
            the work Dong et al. 2021 (problem k3), where phase angle
            modulated bat algorithm (P-AMBA) was proposed for binary problems
            with application to antenna topology optimization.
        """)

        i = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
             0, 0, 0, 0, 1, 0, 0, 0]
        self.set_min(i=i, y=-3103.)

        self.set_constr(penalty=0.)

    @property
    def args_constr(self):
        return {'d': 50, 'n': 2}

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.zeros(50, dtype=int)
        for k in [0, 12, 34, 44, 45, 47]:
            i[k] = 1
        return np.array(i, dtype=int), -444.0

    @property
    def with_constr(self):
        return True

    def constr(self, i):
        return np.dot(self._w, i) - self._C

    def prep_bm(self):
        self._w = np.array([
            80, 82, 85, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40, 48, 59, 32, 22,
            60, 30, 32, 40, 38, 35, 32, 25, 28, 30, 22, 50, 30, 45, 30, 60, 50,
            20, 65, 20, 25, 30, 10, 20, 25, 15, 10, 10, 10, 4, 4, 2, 1])

        self._p = np.array([
            220, 208, 198, 192, 180, 180, 165, 162, 160, 158, 155, 130, 125,
            122, 120, 118, 115, 110, 105, 101, 100, 100, 98, 96, 95, 90, 88, 82,
            80, 77, 75, 73, 72, 70, 69, 66, 65, 63, 60, 58, 56, 50, 30, 20, 15,
            10, 8, 5, 3, 1])

        self._C = 1000

    def target(self, i):
        return -np.dot(self._p, i)
