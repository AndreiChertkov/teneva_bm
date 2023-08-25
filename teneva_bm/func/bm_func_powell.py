import numpy as np
from teneva_bm.func.func import Func


class BmFuncPowell(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Powell function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-1, 1] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("93. Powell Function"; Continuous, Differentiable,
            Separable, Scalable, Unimodal).
        """)

        self.set_grid(-1., +1., sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 2., 'dy_max': 1.}

    @property
    def ref(self):
        return self.ref_i, 1.979712122400335

    def target_batch(self, X):
        i = np.arange(1, self.d+1)
        return np.sum(np.abs(X)**(i+1), axis=1)
