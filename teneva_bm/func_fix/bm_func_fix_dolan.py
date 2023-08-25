import numpy as np
from teneva_bm.func_fix.func_fix import FuncFix


class BmFuncFixDolan(FuncFix):
    def __init__(self, d=5, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Dolan function (continuous).
            The dimension is 5 and the mode size may be any (default is n=16),
            Default grid limits are [-100, 100] (with small shift);
            the exact global minimum value is known: y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("49. Dolan Function"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
        """)

        self.set_grid(-100., 100., sh=True, sh_out=True)

        self.set_min(y=0.)

    @property
    def args_constr(self):
        return {'d': 5}

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14]
        return np.array(i, dtype=int), 2016.3810305447219

    def target(self, x):
        x1, x2, x3, x4, x5 = x

        y1 = +(x1 + 1.7 * x2) * np.sin(x1)
        y2 = -1.5 * x3
        y3 = -0.1 * x4 * np.cos(x4 + x5 - x1)
        y4 = +0.2 * x5**2
        y5 = -x2
        y6 = -1.

        return y1 + y2 + y3 + y4 + y5 + y6
