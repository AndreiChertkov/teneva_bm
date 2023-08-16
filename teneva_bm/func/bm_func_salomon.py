import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncSalomon(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Salomon function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-100, 100] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("110. Salomon Function"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
        """)

        self.set_grid(-100., +100., sh=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 19.77395062827738

    def target_batch(self, X):
        z = np.sqrt(np.sum(X**2, axis=1))
        return 1. - np.cos(2. * np.pi * z) + 0.1 * z
