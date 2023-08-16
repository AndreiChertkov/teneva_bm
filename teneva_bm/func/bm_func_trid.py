import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncTrid(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Trid function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-d^2, d^2] (with small random shift);
            the exact global minimum is known: xi = i (d+1-i),
            y = -d (d+4) (d-1) / 6.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("149. Trid Function 6"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
            See also https://www.sfu.ca/~ssurjano/trid.html for details.
        """)

        self.set_grid(-self.d**2, +self.d**2, sh=True, sh_out=True)

        i = np.arange(1, self.d+1)
        x = i * (self.d + 1 - i)
        y = -self.d * (self.d + 4) * (self.d - 1) / 6.
        self.set_min(x=x, y=y)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 8556.719852394492

    def target_batch(self, X):
        y1 = np.sum((X-1)**2, axis=1)
        y2 = np.sum(X[:, 1:] * X[:, :-1], axis=1)
        return y1 - y2