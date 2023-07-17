import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Dixon function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    Default grid limits are [-10, 10]; the exact global minimum
    is known: x_i = 2^{(2^i-2) / 2^i} (i = 1, ..., d), y = 0.  Note that
    this function achieves a global minimum at more than one point.
    See https://www.sfu.ca/~ssurjano/dixonpr.html for details.
    See also the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("48. Dixon & Price Function"; Continuous, Differentiable,
    Non-Separable, Scalable, Unimodal).
"""


class BmFuncDixon(Bm):
    def __init__(self, d=50, n=15, name='FuncDixon', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-10., +10.)

        x_min = [1.]
        for _ in range(d-1): # TODO: check this formula one more time:
            x_min.append(np.sqrt(x_min[-1]/2.))
        self.set_min(x=np.array(x_min), y=0.)

    @property
    def is_func(self):
        return True

    def _f_batch(self, X):
        y1 = (X[:, 0] - 1)**2

        y2 = np.arange(2, self.d+1) * (2. * X[:, 1:]**2 - X[:, :-1])**2
        y2 = np.sum(y2, axis=1)

        return y1 + y2

    def _f_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)

        y1 = (x[0] - 1)**2
        y2 = torch.arange(2, d+1) * (2. * x[1:]**2 - x[:-1])**2
        y2 = torch.sum(y2)
        return y1 + y2


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncDixon().prep()
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
