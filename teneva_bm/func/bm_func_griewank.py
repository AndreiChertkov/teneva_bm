import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Griewank function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    Default grid limits are [-100, 100]; the exact global minimum
    is known: x = [0, ..., 0], y = 0.
    See https://www.sfu.ca/~ssurjano/griewank.html for details.
    See also the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("59. Griewank Function"; Continuous, Differentiable, Non-Separable,
    Scalable, Multimodal).
    Note that the method "build_cores" for construction of the function
    in the TT-format on the discrete grid is available.
"""


class BmFuncGriewank(Bm):
    def __init__(self, d=50, n=15, name='FuncGriewank', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-100., +100.)

        self.set_min(
            i=np.array((self.n-1)/2, dtype=int) if self.is_n_odd else None,
            x=[0.]*self.d,
            y=0.)

        self.with_cores = True

    @property
    def is_func(self):
        return True

    def _cores(self, X):
        Y = self._cores_mul([np.cos(x/np.sqrt(i)) for i,x in enumerate(X.T,1)])
        Y[-1] *= -1
        return teneva.add(Y, self._cores_add([x**2 / 4000. for x in X.T], a0=1))

    def _f_batch(self, X):
        y1 = np.sum(X**2, axis=1) / 4000

        y2 = np.cos(X / np.sqrt(np.arange(self.d) + 1))
        y2 = - np.prod(y2, axis=1)

        y3 = 1.

        return y1 + y2 + y3

    def _f_pt(self, x):
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
