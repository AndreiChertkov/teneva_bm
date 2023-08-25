import numpy as np
from teneva_bm.func.func import Func


class BmFuncSchaffer(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Schaffer function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-100, 100] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("135. Schaffer Function F6"; Continuous, Differentiable,
            Non-Separable, Scalable, Multimodal).
            Note that in some sources (see, e.g., "Hybrid genetic deflated
            Newton method for global optimisation") this function has a
            different form (term "x_{i+1}" is missing), but we use the
            analytical formula from the above work.
        """)

        self.set_grid(-100., +100., sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 3., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 2.9876312490738646

    def target_batch(self, X):
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1)

    def _target_pt(self, x):
        """Draft."""
        z = x[:-1]**2 + x[1:]**2
        y = 0.5 + (torch.sin(torch.sqrt(z))**2 - 0.5) / (1. + 0.001 * z)**2
        return torch.sum(y)
