import numpy as np
import teneva
from teneva_bm.func.func import Func


class BmFuncYang(Func):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Wavy function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-2 pi, 2 pi] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("169. Xin-She Yang Function 2"; Discontinuous, Non-differentiable,
            Non-separable, Scalable, Multimodal).
        """)

        self.set_grid(-2.*np.pi, +2.*np.pi, sh=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 85.15511376128357

    def target_batch(self, X):
        y1 = np.sum(np.abs(X), axis=1)
        y2 = np.exp(-np.sum(np.sin(X**2), axis=1))
        return y1 * y2
