import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncAlpine(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Alpine function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-10, 10] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("6. Alpine 1 Function"; Continuous, Non-Differentiable, Separable,
            Non-Scalable, Multimodal).
        """)

        self.set_grid(-10., +10., sh=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 32.67394403036597

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        return self.cores_add([np.abs(x * (np.sin(x) + 0.1)) for x in X.T])

    def target_batch(self, X):
        return np.sum(np.abs(X * np.sin(X) + 0.1 * X), axis=1)

    def _target_pt(self, x):
        """Draft."""
        return torch.sum(torch.abs(x * torch.sin(x) + 0.1 * x))
