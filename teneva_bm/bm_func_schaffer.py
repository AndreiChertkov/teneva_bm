import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Schaffer function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    See the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("136. Schaffer F6 Function"; Continuous, Differentiable,
    Non-Separable, Scalable, Multimodal).
"""


class BmFuncSchaffer(Bm):
    def __init__(self, d=50, n=15, name='FuncSchaffer', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-100., +100.)

        self.set_min(
            i=np.array((self.n-1)/2, dtype=int) if self.is_n_odd else None,
            x=[0.]*self.d,
            y=0.)

    @property
    def is_func(self):
        return True

    def _f_batch(self, X):
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1)

    def _f_pt(self, x):
        """Draft."""
        z = x[:-1]**2 + x[1:]**2
        y = 0.5 + (torch.sin(torch.sqrt(z))**2 - 0.5) / (1. + 0.001 * z)**2
        return torch.sum(y)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncSchaffer().prep()
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

    text = 'Value at minimum (real vs calc)   :  '
    y_real = bm.y_min_real
    y_calc = bm(bm.x_min_real)
    text += f'{y_real:-10.3e}/ {y_calc:-10.3e}'
    print(text)
