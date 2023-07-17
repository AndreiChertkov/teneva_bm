import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    The function 006 from the Hock & Schittkowski collection
    with the explicit constraint.
    The dimension should be 2, and the mode size may be any (default is 21),
    the default limits for function inputs are [-10, 10].
    The exact global minimum is known: x = [1, 1], y = 0.
"""


class BmHsFunc006(Bm):
    def __init__(self, d=2, n=21, name='HsFunc006', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-10., +10.)

        self.set_min(
            x=[1.]*self.d,
            y=0.)

        self.opt_eps = 1.E-16
        self.penalty_constr = 1.E+42
        self.with_constr = True

    @property
    def is_func(self):
        return True

    def _constr_batch(self, X):
        constr = 10. * (X[:, 1] - X[:, 0]**2)
        return np.abs(constr) < self.opt_eps

    def _f_batch(self, X):
        return (1. - X[:, 0])**2


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmHsFunc006().prep()
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

    text = 'Value at a valid point                   :  '
    x = [2., 4.+1.E-18]
    y = bm(x)
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at an invalid point                :  '
    x = [3., 4.]
    y = bm(x)
    text += f'{y:-10.3e}'
    print(text)
