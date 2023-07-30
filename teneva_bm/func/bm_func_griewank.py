import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Griewank function (continuous).
    The dimension and mode size may be any (default are d=7, n=16).
    Default grid limits are [-100, 100] (with small random shift);
    the exact global minimum is known: x = [0, ..., 0], y = 0.
    See the the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("59. Griewank Function"; Continuous, Differentiable, Non-Separable,
    Scalable, Multimodal).
    See also https://www.sfu.ca/~ssurjano/griewank.html for details.
"""


class BmFuncGriewank(Bm):
    def __init__(self, d=7, n=16, name='FuncGriewank', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-100., +100.)
        self.shift_grid()

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Y = self.cores_mul([np.cos(x/np.sqrt(i)) for i,x in enumerate(X.T,1)])
        Y[-1] *= -1
        return teneva.add(Y, self.cores_add([x**2 / 4000. for x in X.T], a0=1))

    def target_batch(self, X):
        y1 = np.sum(X**2, axis=1) / 4000

        y2 = np.cos(X / np.sqrt(np.arange(self.d) + 1))
        y2 = - np.prod(y2, axis=1)

        y3 = 1.

        return y1 + y2 + y3

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)

        y1 = torch.sum(x**2) / 4000

        y2 = torch.cos(x / torch.sqrt(torch.arange(d) + 1.))
        y2 = - torch.prod(y2)

        y3 = 1.

        return y1 + y2 + y3


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncGriewank().prep()
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

    text = 'TT-cores accuracy on train data          :  '
    Y = bm.build_cores()
    e = teneva.accuracy_on_data(Y, I_trn, y_trn)
    text += f'{e:-10.1e}'
    print(text)

    text = 'Value at the minimum (real vs calc)      :  '
    y_real = bm.y_min_real
    y_calc = bm(bm.x_min_real)
    text += f'{y_real:-10.3e}       /      {y_calc:-10.3e}'
    print(text)
