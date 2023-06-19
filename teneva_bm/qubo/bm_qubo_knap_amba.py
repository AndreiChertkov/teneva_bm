import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Binary knapsack problem with fixed weights wi in [5, 20], profits pi in
    [50, 100] (i = 1, 2, . . . , d) and the maximum capacity C = 1000. It is
    from work (Dong et al., 2021) (problem k3; d = 50), where anglemodulated
    bat algorithm (AMBA) algorithm was proposed for high-dimensional binary
    optimization problems with application to antenna topology optimization.
    The dimension should be 50, and the mode size should be 2.
"""


class BmQuboKnapAmba(Bm):
    def __init__(self, d=50, n=2, name='QuboKnapAmba', desc=DESC):
        super().__init__(d, n, name, desc)

        if self.d != 50:
            self.set_err('Dimension should be "50"')
        if not self.is_n_equal or self.n[0] != 2:
            self.set_err('Mode size (n) should be "2"')

        self.set_min(i=np.array([
            1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0,
            1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=int),
            y=-3103.)

    @property
    def is_tens(self):
        return True

    def prep(self):
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

        self._is_prep = True
        return self

    def _f(self, i):
        cost = np.dot(self.bm_p, i)
        constr = np.dot(self.bm_w, i)
        return 0 if constr > self.bm_C else -cost


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmQuboKnapAmba().prep()
    print(bm.info())

    text = 'Range of y for 10K random samples : '
    bm.build_trn(1.E+4)
    text += f'[{np.min(bm.y_trn):-10.3e},'
    text += f' {np.max(bm.y_trn):-10.3e}] '
    text += f'(avg: {np.mean(bm.y_trn):-10.3e})'
    print(text)

    text = 'Value at a random multi-index     :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices   :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)

    text = 'Value at minimum (real vs calc)   :  '
    y_real = bm.y_min_real
    y_calc = bm[bm.i_min_real]
    text += f'{y_real:-10.3e}/ {y_calc:-10.3e}'
    print(text)
