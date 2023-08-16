import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncTrigonometric(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Trigonometric function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [0, pi] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("152. Trigonometric Function 1"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
        """)

        self.set_grid(0., np.pi, sh=True, sh_out=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 543.0253567898842

    def target_batch(self, X):
        i = np.arange(1, self.d+1)

        y1 = self.d
        y2 = -np.sum(np.cos(X), axis=1)
        Y2 = np.hstack([y2.reshape(-1, 1)]*self.d)
        Y3 = i * (1. - np.cos(X) - np.sin(X))

        return np.sum((y1 + Y2 + Y3)**2, axis=1)
