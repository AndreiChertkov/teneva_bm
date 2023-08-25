import numpy as np
from teneva_bm import Bm


class BmQuboFixKnap80(Bm):
    def __init__(self, d=80, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Binary 80-dimensional knapsack problem
            -1 * sum p_i x_i -> min s.t. sum w_i x_i < C
            with fixed weights w, profits p and the capacity C.
            The exact minimum is known (-5183). We use the parameters from
            the work Dong et al. 2021 (problem k4), where phase angle
            modulated bat algorithm (P-AMBA) was proposed for binary problems
            with application to antenna topology optimization.
        """)

        i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
             1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.set_min(i=i, y=-5183)

        self.set_constr(penalty=0.)

    @property
    def args_constr(self):
        return {'d': 80, 'n': 2}

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.zeros(80, dtype=int)
        for k in [0, 12, 34, 44, 53, 65, 77]:
            i[k] = 1
        return np.array(i, dtype=int), -611.0

    @property
    def with_constr(self):
        return True

    def constr(self, i):
        return np.dot(self._w, i) - self._C

    def prep_bm(self):
        self._w = np.array([
            40, 27, 5, 21, 51, 16, 42, 18, 52, 28, 57, 34, 44, 43, 52, 55,
            53, 42, 47, 56, 57, 44, 16, 2, 12, 9, 40, 23, 56, 3, 39, 16,
            54, 36, 52, 5, 53, 48, 23, 47, 41, 49, 22, 42, 10, 16, 53, 58,
            40, 1, 43, 56, 40, 32, 44, 35, 37, 45, 52, 56, 40, 2, 23,49, 50,
            26, 11, 35, 32, 34, 58, 6, 52, 26, 31, 23, 4, 52, 53, 19])

        self._p = np.array([
            199, 194, 193, 191, 189, 178, 174, 169, 164, 164, 161, 158, 157,
            154, 152, 152, 149, 142, 131, 125, 124, 124, 124, 122, 119, 116,
            114, 113, 111, 110, 109, 100, 97, 94, 91, 82, 82, 81, 80, 80, 80,
            79, 77, 76, 74, 72, 71, 70, 69,68, 65, 65, 61, 56, 55, 54, 53, 47,
            47, 46, 41, 36, 34, 32, 32,30, 29, 29, 26, 25, 23, 22, 20, 11, 10,
            9, 5, 4, 3, 1])

        self._C = 1173

    def target(self, i):
        return -np.dot(self._p, i)
