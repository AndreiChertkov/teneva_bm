import numpy as np
from teneva_bm import Bm


class BmQuboFixKnap100(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Binary 100-dimensional knapsack problem
            -1 * sum p_i x_i -> min s.t. sum w_i x_i < C
            with fixed weights w, profits p and the capacity C.
            The exact minimum is known (-15170). We use the parameters from
            the work Dong et al. 2021 (problem k5), where phase angle
            modulated bat algorithm (P-AMBA) was proposed for binary problems
            with application to antenna topology optimization.
        """)

        i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
             1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        self.set_min(i=i, y=-15170)

        self.set_constr(penalty=0.)

    @property
    def args_constr(self):
        return {'d': 100, 'n': 2}

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.zeros(100, dtype=int)
        for k in [0, 12, 34, 44, 53, 65, 99]:
            i[k] = 1
        return np.array(i, dtype=int), -1249.0

    @property
    def with_constr(self):
        return True

    def constr(self, i):
        return np.dot(self._w, i) - self._C

    def prep_bm(self):
        self._w = np.array([
            54, 95, 36, 18, 4, 71, 83, 16, 27, 84, 88, 45, 94, 64, 14, 80,
            4, 23, 75, 36, 90, 20, 77, 32, 58, 6, 14, 86, 84, 59, 71, 21, 30,
            22, 96, 49, 81, 48, 37, 28, 6, 84, 19, 55, 88, 38, 51, 52, 79, 55,
            70, 53, 64, 99, 61, 86, 1, 64, 32, 60, 42, 45, 34, 22, 49, 37, 33,
            1, 78, 43, 85, 24, 96, 32, 99, 57, 23, 8, 10, 74, 59, 89, 95, 40,
            46, 65, 6, 89, 84, 83, 6, 19, 45, 59, 26, 13, 8, 26, 5, 9])

        self._p = np.array([
            297, 295, 293, 292, 291, 289, 284, 284, 283, 283, 281, 280, 279,
            277, 276, 275, 273,264, 260, 257, 250, 236, 236, 235, 235, 233,
            232, 232, 228, 218, 217, 214, 211, 208, 205, 204, 203, 201, 196,
            194, 193, 193, 192, 191, 190, 187, 187, 184, 184, 184, 181, 179,
            176, 173, 172, 171, 160, 128, 123, 114, 113, 107, 105, 101, 100,
            100, 99, 98, 97, 94, 94, 93, 91, 80, 74, 73, 72, 63, 63, 62, 61,
            60, 56, 53, 52, 50, 48, 46, 40, 40, 35, 28, 22, 22, 18, 15, 12,
            11, 6, 5])

        self._C = 3818

    def target(self, i):
        return -np.dot(self._p, i)
