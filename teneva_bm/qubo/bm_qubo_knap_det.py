import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Binary knapsack problem -1 * sum p_i x_i -> min s.t. sum w_i x_i < C with
    fixed weights w, profits p and the capacity C. The exact minimum is known
    (d=10: -295, d=20: -1024, d=50: -3103, d=80: -5183, d=100: -15170).
    We use the values of parameters from the work Dong et al. 2021 (problems
    k1-k5), where phase angle modulated bat algorithm (P-AMBA) was proposed
    for high dimensional binary optimization problems with application to
    antenna topology optimization. The dimension should be in 10, 20, 50, 80,
    100, and the mode size should be 2. Note that the default penalty for the
    constraint is "0".
"""


class BmQuboKnapDet(Bm):
    def __init__(self, d=50, n=2, name='QuboKnapDet', desc=DESC):
        super().__init__(d, n, name, desc)

        if not self.is_n_equal or self.n0 != 2:
            self.set_err('Mode size (n) should be "2"')

        if self.d == 10:
            i = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
            self.set_min(i=i, y=-295)
        elif self.d == 20:
            i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
            self.set_min(i=i, y=-1024)
        elif self.d == 50:
            i = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
                 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
                 0, 0, 0, 0, 1, 0, 0, 0]
            self.set_min(i=i, y=-3103.)
        elif self.d == 80:
            i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
                 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.set_min(i=i, y=-5183)
        elif self.d == 100:
            i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
            self.set_min(i=i, y=-15170)
        else:
            self.set_err('Dimension should be in 10, 20, 50, 80, 100')

        self.set_constr(penalty=0.)

    @property
    def is_tens(self):
        return True

    @property
    def with_constr(self):
        return True

    def prep(self):
        self.check_err()

        if self.d == 10:
            self.prep_10()
        elif self.d == 20:
            self.prep_20()
        elif self.d == 50:
            self.prep_50()
        elif self.d == 80:
            self.prep_80()
        elif self.d == 100:
            self.prep_100()

        self.bm_w = np.array(self.bm_w)
        self.bm_p = np.array(self.bm_p)

        self.is_prep = True
        return self

    def prep_10(self):
        self.bm_w = [
            95, 4, 60, 32, 23, 72, 80, 62, 65, 46]

        self.bm_p = [
            55, 10, 47, 5, 4, 50, 8, 61, 85, 87]

        self.bm_C = 269

    def prep_20(self):
        self.bm_w = [
            92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70,
            48, 14, 58]

        self.bm_p = [
            44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17, 75,
            29, 75, 63]

        self.bm_C = 878

    def prep_50(self):
        self.bm_w = [
            80, 82, 85, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40, 48, 59, 32, 22,
            60, 30, 32, 40, 38, 35, 32, 25, 28, 30, 22, 50, 30, 45, 30, 60, 50,
            20, 65, 20, 25, 30, 10, 20, 25, 15, 10, 10, 10, 4, 4, 2, 1]

        self.bm_p = [
            220, 208, 198, 192, 180, 180, 165, 162, 160, 158, 155, 130, 125,
            122, 120, 118, 115, 110, 105, 101, 100, 100, 98, 96, 95, 90, 88, 82,
            80, 77, 75, 73, 72, 70, 69, 66, 65, 63, 60, 58, 56, 50, 30, 20, 15,
            10, 8, 5, 3, 1]

        self.bm_C = 1000

    def prep_80(self):
        self.bm_w = [
            40, 27, 5, 21, 51, 16, 42, 18, 52, 28, 57, 34, 44, 43, 52, 55,
            53, 42, 47, 56, 57, 44, 16, 2, 12, 9, 40, 23, 56, 3, 39, 16,
            54, 36, 52, 5, 53, 48, 23, 47, 41, 49, 22, 42, 10, 16, 53, 58,
            40, 1, 43, 56, 40, 32, 44, 35, 37, 45, 52, 56, 40, 2, 23,49, 50,
            26, 11, 35, 32, 34, 58, 6, 52, 26, 31, 23, 4, 52, 53, 19]

        self.bm_p = [
            199, 194, 193, 191, 189, 178, 174, 169, 164, 164, 161, 158, 157,
            154, 152, 152, 149, 142, 131, 125, 124, 124, 124, 122, 119, 116,
            114, 113, 111, 110, 109, 100, 97, 94, 91, 82, 82, 81, 80, 80, 80,
            79, 77, 76, 74, 72, 71, 70, 69,68, 65, 65, 61, 56, 55, 54, 53, 47,
            47, 46, 41, 36, 34, 32, 32,30, 29, 29, 26, 25, 23, 22, 20, 11, 10,
            9, 5, 4, 3, 1]

        self.bm_C = 1173

    def prep_100(self):
        self.bm_w = [
            54, 95, 36, 18, 4, 71, 83, 16, 27, 84, 88, 45, 94, 64, 14, 80,
            4, 23, 75, 36, 90, 20, 77, 32, 58, 6, 14, 86, 84, 59, 71, 21, 30,
            22, 96, 49, 81, 48, 37, 28, 6, 84, 19, 55, 88, 38, 51, 52, 79, 55,
            70, 53, 64, 99, 61, 86, 1, 64, 32, 60, 42, 45, 34, 22, 49, 37, 33,
            1, 78, 43, 85, 24, 96, 32, 99, 57, 23, 8, 10, 74, 59, 89, 95, 40,
            46, 65, 6, 89, 84, 83, 6, 19, 45, 59, 26, 13, 8, 26, 5, 9]

        self.bm_p = [
            297, 295, 293, 292, 291, 289, 284, 284, 283, 283, 281, 280, 279,
            277, 276, 275, 273,264, 260, 257, 250, 236, 236, 235, 235, 233,
            232, 232, 228, 218, 217, 214, 211, 208, 205, 204, 203, 201, 196,
            194, 193, 193, 192, 191, 190, 187, 187, 184, 184, 184, 181, 179,
            176, 173, 172, 171, 160, 128, 123, 114, 113, 107, 105, 101, 100,
            100, 99, 98, 97, 94, 94, 93, 91, 80, 74, 73, 72, 63, 63, 62, 61,
            60, 56, 53, 52, 50, 48, 46, 40, 40, 35, 28, 22, 22, 18, 15, 12,
            11, 6, 5]

        self.bm_C = 3818

    def _c(self, i):
        return np.dot(self.bm_w, i) - self.bm_C

    def _f(self, i):
        return -np.dot(self.bm_p, i)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmQuboKnapDet().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+4)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices          :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)

    for d in [10, 20, 50, 80, 100]:
        text = f'Value at the minimum for d={d:-3d}           :  '
        bm = BmQuboKnapDet(d).prep()
        y_real = bm.y_min_real
        y_calc = bm[bm.i_min_real]
        text += f'{y_real:-10.3e}       /      {y_calc:-10.3e}'
        print(text)
