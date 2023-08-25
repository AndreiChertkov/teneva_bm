import numpy as np
from teneva_bm.func.func import Func


class BmFuncDixon(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Dixon function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-10, 10] (with small random shift);
            the exact global minimum is known: x_i = 2^{(2^i-2) / 2^i}
            (i = 1, ..., d), y = 0;  note that this function achieves a global
            minimum at more than one point.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("48. Dixon & Price Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Unimodal).
            See also https://www.sfu.ca/~ssurjano/dixonpr.html for details.
        """)

        self.set_grid(-10., +10., sh=True)

        x = [1.]
        for _ in range(d-1):
            x.append(np.sqrt(x[-1] / 2.))
        self.set_min(x=x, y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 1.E+5, 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 383674.42504801514

    def target_batch(self, X):
        y1 = (X[:, 0] - 1)**2

        y2 = np.arange(2, self.d+1) * (2. * X[:, 1:]**2 - X[:, :-1])**2
        y2 = np.sum(y2, axis=1)

        return y1 + y2

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)

        y1 = (x[0] - 1)**2

        y2 = torch.arange(2, d+1) * (2. * x[1:]**2 - x[:-1])**2
        y2 = torch.sum(y2)

        return y1 + y2
