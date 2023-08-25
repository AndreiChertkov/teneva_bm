import numpy as np
from teneva_bm import Bm


class BmQuboFixKnap10(Bm):
    def __init__(self, d=10, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Binary 10-dimensional knapsack problem
            -1 * sum p_i x_i -> min s.t. sum w_i x_i < C
            with fixed weights w, profits p and the capacity C.
            The exact minimum is known (-295). We use the parameters from
            the work Dong et al. 2021 (problem k1), where phase angle
            modulated bat algorithm (P-AMBA) was proposed for binary problems
            with application to antenna topology optimization.
        """)

        i = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        self.set_min(i=i, y=-295)

        self.set_constr(penalty=0.)

    @property
    def args_constr(self):
        return {'d': 10, 'n': 2}

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.zeros(10, dtype=int)
        for k in [0, 2, 5]:
            i[k] = 1
        return np.array(i, dtype=int), -152.0

    @property
    def with_constr(self):
        return True

    def constr(self, i):
        return np.dot(self._w, i) - self._C

    def prep_bm(self):
        self._w = np.array([
            95, 4, 60, 32, 23, 72, 80, 62, 65, 46])

        self._p = np.array([
            55, 10, 47, 5, 4, 50, 8, 61, 85, 87])

        self._C = 269

    def target(self, i):
        return -np.dot(self._p, i)
