import numpy as np
from teneva_bm.func.func import Func


class BmFuncChung(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Chung Reynolds function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-10, 10] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            Note that we have specified smaller limits for argument changes
            relative to the commonly used [-100, 100].
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("34. Chung Reynolds Function"; Continuous, Differentiable,
            Partially-separable, Scalable, Unimodal).
        """)

        self.set_grid(-10., +10., sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 1.E+5, 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 105869.01898168476

    def target_batch(self, X):
        return np.sum(X**2, axis=1)**2
