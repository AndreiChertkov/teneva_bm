import numpy as np
from teneva_bm.func.func import Func


class BmFuncTrigonometric(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

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

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 25., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 543.0253567898842

    def target_batch(self, X):
        i = np.arange(1, self.d+1)

        y1 = self.d
        y2 = -np.sum(np.cos(X), axis=1)
        Y2 = np.hstack([y2.reshape(-1, 1)]*self.d)
        Y3 = i * (1. - np.cos(X) - np.sin(X))

        return np.sum((y1 + Y2 + Y3)**2, axis=1)
