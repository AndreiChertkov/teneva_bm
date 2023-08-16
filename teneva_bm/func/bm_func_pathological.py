import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncPathological(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Pathological function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-100, 100] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("87. Pathological Function"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
        """)

        self.set_grid(-100., +100., sh=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 2.9992923549128307

    def target_batch(self, X):
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        y1 = np.sin(np.sqrt(100. * X1**2 + X2**2))**2 - 0.5
        y2 = 1. + 0.001 * (X1**2 - 2 * X1 * X2 + X2**2)**2
        return np.sum(0.5 + y1 / y2, axis=1)
