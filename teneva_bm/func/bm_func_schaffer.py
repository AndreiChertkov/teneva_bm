import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncSchaffer(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Schaffer function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-100, 100] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("135. Schaffer F6 Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Multimodal).
        """)

        self.set_grid(-100., +100., sh=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 2.9876312490738646

    def target_batch(self, X):
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1)

    def _target_pt(self, x):
        """Draft."""
        z = x[:-1]**2 + x[1:]**2
        y = 0.5 + (torch.sin(torch.sqrt(z))**2 - 0.5) / (1. + 0.001 * z)**2
        return torch.sum(y)
