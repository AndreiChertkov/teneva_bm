import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Continuous optimal control (OC) problem:
    .-------------.
    | F(i) -> min |
    .-------------.
    i - integer control
        i[0] = 0 | >= -1.5 | <= 4
        i[1] = 0 | >= -3   | <= 3
    F - objective function
        sin(i[0] + i[1]) + (i[0] - i[1])^2 - 1.5*i[0] + 2.5*i[1] + 1
    The best value is (-1)*sqrt(3)/2 - pi/3 ~= -1.91 
    and the corresponding solution is 
    i[0] = (-1)*pi/3 + 1/2 ~= -0.55
    i[1] = (-1)*pi/3 - 1/2 ~= -1.54
    The dimension d is 2, and the mode size is 21.
"""


class BmHS005(Bm):
    def __init__(self, d=3, n=21, name='hs5', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(
            [-1.5, -3, -10], 
            [4, 3, 10]
        )

        self.set_min(
            i=None,
            x=np.array([
                -np.pi/3 + 1/2,
                -np.pi/3 - 1/2, 
                0.0]),
            y=-np.sqrt(3)/2-np.pi/3
        )

    @property
    def is_func(self):
        return True

    def _f_batch(self, X):
        y1 = np.sin(X[:, 0] + X[:, 1])
        y2 = (X[:, 0] - X[:, 1]) ** 2
        y3 = -1.5 * X[:, 0] + 2.5 * X[:, 1] + 1
        y = y1 + y2 + y3
        return y


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmHS005().prep()
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