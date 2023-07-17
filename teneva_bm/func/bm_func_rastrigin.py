import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Rastrigin function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    Default grid limits are [-5.12, 5.12]; the exact global minimum
    is known: x = [0, ..., 0], y = 0.
    See https://www.sfu.ca/~ssurjano/rastr.html for details.
    See also the work Johannes M Dieterich, Bernd Hartke. "Empirical review
    of standard benchmark functions using evolutionary global optimization".
    Applied Mathematics 2012; 3:1552-1564.
    Note that the method "build_cores" for construction of the function
    in the TT-format on the discrete grid is available.
"""


class BmFuncRastrigin(Bm):
    def __init__(self, d=50, n=15, name='FuncRastrigin', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-5.12, +5.12)

        self.set_min(
            i=np.array((self.n-1)/2, dtype=int) if self.is_n_odd else None,
            x=[0.]*self.d,
            y=0.)

        self.with_cores = True

    @property
    def is_func(self):
        return True

    def set_opts(self, A=10.):
        """Setting options specific to this benchmark.

        Args:
            A (float): parameter of the function.

        """
        self.opt_A = A

    def _cores(self, X):
        return self._cores_add(
            [x**2 - self.opt_A * np.cos(2 * np.pi * x) for x in X.T],
            a0=self.opt_A*self.d)

    def _f_batch(self, X):
        y1 = self.opt_A * self.d
        y2 = np.sum(X**2 - self.opt_A * np.cos(2. * np.pi * X), axis=1)
        return y1 + y2

    def _f_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        par_A = torch.tensor(self.opt_A)
        pi = torch.tensor(np.pi)

        y1 = par_A * d
        y2 = torch.sum(x**2 - par_A * torch.cos(2. * pi * x))

        return y1 + y2


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncRastrigin().prep()
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
