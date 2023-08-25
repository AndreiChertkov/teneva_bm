import numpy as np
from teneva_bm.func.func import Func


class BmFuncExp(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Exponential function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-1, 1] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = -1.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("54. Exponential Function"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
        """)

        self.set_grid(-1., +1., sh=True)

        self.set_min(x=0., y=-1.)

    @property
    def opts_plot(self):
        return {'dy_min': 0.5, 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, -0.19654261789623134

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Y = self.cores_mul([np.exp(-0.5 * x**2) for x in X.T])
        Y[-1] *= -1.
        return Y

    def target_batch(self, X):
        return -np.exp(-0.5 * np.sum(X**2, axis=1))

    def _target_pt(self, x):
        """Draft."""
        return -torch.exp(-0.5 * torch.sum(x**2))
