import numpy as np
from teneva_bm.func.func import Func


class BmFuncSchwefel(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Schwefel function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [0, 500] (with small random shift; note
            also that we limit this function to this domain instead of often
            used [-500, 500] to make sure it has a single global minimum).
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("127. Schwefel Function 2.26"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
        """)

        self.set_grid(0., +500., sh=True)

    @property
    def opts_plot(self):
        return {'dy_min': 120., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, -112.91878452976069

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        return self.cores_add(
            [-x * np.sin(np.sqrt(np.abs(x))) / self.d for x in X.T], a0=0.)

    def target_batch(self, X):
        return -np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1) / self.d
