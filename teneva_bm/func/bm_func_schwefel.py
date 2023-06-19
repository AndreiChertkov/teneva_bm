import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Schwefel function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    See https://www.sfu.ca/~ssurjano/schwef.html for details.
    See also the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("128. Schwefel 2.26 Function"; Continuous, Differentiable,
    Separable, Scalable, Multimodal).
"""


class BmFuncSchwefel(Bm):
    def __init__(self, d=50, n=15, name='FuncSchwefel', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-500., +500.)

        self.set_min(x=[420.9687]*self.d, y=0.)

        self.with_cores = True

    @property
    def is_func(self):
        return True

    def set_opts(self, a=418.9829):
        """Setting options specific to this benchmark.

        Args:
            a (float): parameter of the function.

        """
        self.opt_a = a

    def _cores(self, X):
        return self._cores_add(
            [-x * np.sin(np.sqrt(np.abs(x))) for x in X.T],
            a0=self.opt_a*self.d)

    def _f_batch(self, X):
        y0 = self.opt_a * self.d
        return y0 - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

    def _f_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        par_a = torch.tensor(self.opt_a)
        y0 = par_a * d
        return y0 - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncSchwefel().prep()
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

    text = 'TT-cores accuracy on train data   :  '
    Y = bm.build_cores()
    e = teneva.accuracy_on_data(Y, bm.I_trn, bm.y_trn)
    text += f'{e:-10.1e}'
    print(text)

    text = 'Value at minimum (real vs calc)   :  '
    y_real = bm.y_min_real
    y_calc = bm(bm.x_min_real)
    text += f'{y_real:-10.3e}/ {y_calc:-10.3e}'
    print(text)
