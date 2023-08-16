import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncPowell(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

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

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 1.979712122400335

    def target_batch(self, X):
        i = np.arange(2, self.d+2)
        return np.sum(np.abs(X)**i, axis=1)
