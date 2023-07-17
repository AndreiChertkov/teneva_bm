import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Ackley function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    Default grid limits are [-32.768, 32.768]; the exact global minimum
    is known: x = [0, ..., 0], y = 0.
    See https://www.sfu.ca/~ssurjano/ackley.html for details.
    See also the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("1. Ackley 1 Function"; Continuous, Differentiable, Non-separable,
    Scalable, Multimodal).
"""


class BmFuncAckley(Bm):
    def __init__(self, d=50, n=15, name='FuncAckley', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-32.768, +32.768)

        self.set_min(
            i=np.array((self.n-1)/2, dtype=int) if self.is_n_odd else None,
            x=[0.]*self.d,
            y=0.)

    @property
    def is_func(self):
        return True

    def set_opts(self, a=20., b=0.2, c=2.*np.pi):
        """Setting options specific to this benchmark.

        Args:
            a (float): parameter of the function.
            b (float): parameter of the function.
            c (float): parameter of the function.

        """
        self.opt_a = a
        self.opt_b = b
        self.opt_c = c

    def _f_batch(self, X):
        y1 = np.sqrt(np.sum(X**2, axis=1) / self.d)
        y1 = -self.opt_a * np.exp(-self.opt_b * y1)

        y2 = np.sum(np.cos(self.opt_c * X), axis=1)
        y2 = -np.exp(y2 / self.d)

        y3 = self.opt_a + np.exp(1.)

        return y1 + y2 + y3

    def _f_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        par_a = torch.tensor(self.opt_a)
        par_b = torch.tensor(self.opt_b)
        par_c = torch.tensor(self.opt_c)

        y1 = torch.sqrt(torch.sum(x**2) / d)
        y1 = - par_a * torch.exp(-par_b * y1)

        y2 = torch.sum(torch.cos(par_c * x))
        y2 = - torch.exp(y2 / d)

        y3 = par_a + torch.exp(torch.tensor(1.))

        return y1 + y2 + y3


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncAckley().prep()
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
