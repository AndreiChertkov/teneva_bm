import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    The function 001 from the Hock & Schittkowski collection.
    The dimension should be 2, and the mode size may be any (default is 21),
    the default limits for function inputs are [-10, 10].
    The exact global minimum is known: x = [1, 1], y = 0.
"""


class BmHsFunc001(Bm):
    def __init__(self, d=2, n=21, name='HsFunc001', desc=DESC):
        super().__init__(d, n, name, desc)

        if self.d != 2:
            self.set_err('Dimension should be 2')

        self.set_grid(-10., +10.)

        self.set_min(
            x=[1.]*self.d,
            y=0.)

    @property
    def is_func(self):
        return True

    def _f_batch(self, X):
        return 100. * (X[:, 1] - X[:, 0]**2)**2 + (1. - X[:, 0])**2


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmHsFunc001().prep()
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
    y_calc = bm(bm.x_min_real)
    text += f'{y_real:-10.3e}       /      {y_calc:-10.3e}'
    print(text)
