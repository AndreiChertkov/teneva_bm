import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Continuous optimal control (OC) problem:
    .-------------.
    | F(i) -> min |
    .-------------.
    i - integer control
        i[0] = -3 | >= -10 | <= 10
        i[1] = -1 | >= -10 | <= 10
        i[2] = -3 | >= -10 | <= 10
        i[3] = -1 | >= -10 | <= 10
    F - objective function
        100 * (i[1] - i[0]^2)^2 + (1 - i[0])^2 +  
        90 * (i[3] - i[2]^2)^2 + (1 - i[2])^2 + 
        10.1 * ((i[1] - 1)^2 + (i[3] - 1)^2) + 
        19.8 * (i[1] - 1) * (i[3] - 1)
    The best value is 0 and the corresponding solution is [1, 1, 1, 1]
    The dimension d is 4, and the mode size is 21.
"""


class BmHS038(Bm):
    def __init__(self, d=4, n=21, name='hs38', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-10, 10)

        self.set_min(
            i=None,
            x=np.array([0]*self.d),
            y=0
        )

    @property
    def is_func(self):
        return True

    def _f_batch(self, X):
        y1 = 100 * (X[:, 1] - X[:, 0] ** 2) ** 2 + (1 - X[:, 0]) ** 2
        y2 = 90 * (X[:, 3] - X[:, 2] ** 2) ** 2 + (1 - X[:, 2]) ** 2
        y3 = 10.1 * ((X[:, 1] - 1) ** 2 + (X[:, 3] - 1) ** 2) 
        y4 = 19.8 * (X[:, 1] - 1) * (X[:, 3] - 1)
        y = y1 + y2 + y3 + y4
        return y


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmHS038().prep()
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
