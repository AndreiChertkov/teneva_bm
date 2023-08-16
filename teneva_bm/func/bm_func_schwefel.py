import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncSchwefel(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Schwefel function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [0, 500] (with small random shift; note
            also that we limit this function to this domain instead of often
            used [-500, 500] to make sure it has a single global minimum);
            the exact global minimum is known: x = 420.9687, y = -418.9829.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("127. Schwefel 2.26 Function"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
        """)

        self.set_grid(0., +500., sh=True)

        self.set_min(x=[420.9687]*self.d, y=-418.9829)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), -112.91878452976069

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        return self.cores_add(
            [-x * np.sin(np.sqrt(np.abs(x))) / self.d for x in X.T], a0=0.)

    def target_batch(self, X):
        return -np.sum(X * np.sin(np.sqrt(X)), axis=1) / self.d
