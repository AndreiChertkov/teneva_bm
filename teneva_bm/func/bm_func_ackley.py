import numpy as np


from teneva_bm import Bm


DESC = """
    Analytical Ackley function (continuous).
    The dimension and mode size may be any (default are d=7, n=16).
    Default grid limits are [-32.768, 32.768] (with small random shift);
    the exact global minimum is known: x = [0, ..., 0], y = 0.
    See the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("1. Ackley 1 Function"; Continuous, Differentiable, Non-separable,
    Scalable, Multimodal).
    See also https://www.sfu.ca/~ssurjano/ackley.html for details.
"""


class BmFuncAckley(Bm):
    def __init__(self, d=7, n=16, name='FuncAckley', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-32.768, +32.768)
        self.shift_grid()

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    def get_config(self):
        conf = super().get_config()
        conf['_a'] = self._a
        conf['_b'] = self._b
        conf['_c'] = self._c
        return conf

    def info(self, footer=''):
        text = ''

        text += 'Param a for Ackley function              : '
        v = self._a
        text += f'{v:.6f}\n'

        text += 'Param b for Ackley function              : '
        v = self._b
        text += f'{v:.6f}\n'

        text += 'Param c for Ackley function              : '
        v = self._c
        text += f'{v:.6f}\n'

        return super().info(text+footer)

    def set_opts(self, a=20., b=0.2, c=2.*np.pi):
        """Set options specific to this benchmark.

        There are no plans to manually change the default values.

        Args:
            a (float): parameter of the function.
            b (float): parameter of the function.
            c (float): parameter of the function.

        """
        self._a = a
        self._b = b
        self._c = c

    def target_batch(self, X):
        y1 = np.sqrt(np.sum(X**2, axis=1) / self.d)
        y1 = -self._a * np.exp(-self._b * y1)

        y2 = np.sum(np.cos(self._c * X), axis=1)
        y2 = -np.exp(y2 / self.d)

        y3 = self._a + np.exp(1.)

        return y1 + y2 + y3

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        par_a = torch.tensor(self._a)
        par_b = torch.tensor(self._b)
        par_c = torch.tensor(self._c)

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
