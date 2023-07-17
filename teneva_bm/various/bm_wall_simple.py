import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Simple example of the special discrete function ("wall"), which is
    difficult to optimize by tensor methods. The exact global minimum
    is known: i = [0, ..., 0], y = 0. The target function returns "0" if
    the requested multi-index is optimal; returns a large number ("10 * d")
    if the requested multi-index matches the optimal one in at least one
    element; and returns the first element of the multi-index otherwise.
    The dimension and mode size may be any (default are d=10, n=50).
"""


class BmWallSimple(Bm):
    def __init__(self, d=10, n=50, name='WallSimple', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_min(
            i=np.zeros(self.d, dtype=int),
            y=0.)

    @property
    def is_tens(self):
        return True

    def _f(self, i):
        if len(np.where(i == self.i_min_real)[0]) == self.d:
            return 0.
        elif len(np.where(i == self.i_min_real)[0]) > 0:
            return self.d * 10
        else:
            return i[0]


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmWallSimple().prep()
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

    text = 'Value at the minimum (real vs calc)      :  '
    y_real = bm.y_min_real
    y_calc = bm[bm.i_min_real]
    text += f'{y_real:-10.3e}       /      {y_calc:-10.3e}'
    print(text)
