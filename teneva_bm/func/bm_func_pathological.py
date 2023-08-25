import numpy as np
from teneva_bm.func.func import Func


class BmFuncPathological(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

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
        """
        )

        self.set_grid(-100., +100., sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 2., 'dy_max': 1.}

    @property
    def ref(self):
        return self.ref_i, 2.9992923549128307

    def target_batch(self, X):
        X1 = X[:, :-1]
        X2 = X[:, 1:]

        Y1 = (np.sin(np.sqrt(100. * X1**2 + X2**2)))**2 - 0.5
        Y2 = 1. + 0.001 * (X1**2 - 2. * X1 * X2 + X2**2)**2

        return np.sum(0.5 + Y1 / Y2, axis=1)
