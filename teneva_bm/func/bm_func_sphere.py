import numpy as np
from teneva_bm.func.func import Func


class BmFuncSphere(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Sphere function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-5.12, 5.12] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("136. Sphere Function"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
            See also https://www.sfu.ca/~ssurjano/spheref.html for details (note
            that we use grid limits from this link).
        """)

        self.set_grid(-5.12, +5.12, sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 100., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 85.29515570637663

    def target_batch(self, X):
        return np.sum(X**2, axis=1)
