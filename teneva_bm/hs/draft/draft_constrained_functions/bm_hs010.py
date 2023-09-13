import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Continuous optimal control (OC) problem with constraints:
    .------------------------------.
    | F(i) -> min s.t. C(i) = True |
    .------------------------------.
    i - integer control
        i[0] = -10 | >= -10 | <= 10
        i[1] = 10  | >= -10 | <= 10
    F - objective function
        i[0] - i[1]
    C - constraints
        -3 * i[0]^2 + 2 * i[0] * i[1] - i[1]^2 >= -1
    The best value is -1 and the corresponding solution is [0, 1]
    The dimension d is 2, and the mode size is 21.
"""


class BmHS010(Bm):
    def __init__(self, d=3, n=21, name='hs10', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-10, 10)

        self.set_min(
            i=None,
            x=np.array([-3, -4, 0]),
            y=-1
        )

    @property
    def is_func(self):
        return True
    
    def bm_constr(self, X):
        y = -3 * X[:, 0] ** 2 + 2 * X[:, 0] * X[:, 1] - X[:, 1] ** 2 >= -1
        return ~y

    def _f_batch_good(self, X):
        y = X[:, 0] - X[:, 1]
        return y

    def _f_batch(self, X):
        mask = self.bm_constr(X)
        y = np.zeros(X.shape[0])
        y[~mask] = 1.E+10
        if any(mask):
            y[mask] = self._f_batch_good(X[mask])
        return y


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmHS010().prep()
    print(bm.info())

    text = 'Range of y for 10 random samples : '
    bm.build_trn(1.E+1)
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
    y_calc = bm(bm.x_min_real)
    text += f'{y_real:-10.3e}/ {y_calc:-10.3e}'
    print(text)
