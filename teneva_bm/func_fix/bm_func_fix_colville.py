import numpy as np
from teneva_bm.func_fix.func_fix import FuncFix


class BmFuncFixColville(FuncFix):
    def __init__(self, d=4, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Colville function (continuous).
            The dimension is 4 and the mode size may be any (default is n=16),
            Default grid limits are [-10, 10] (with small shift);
            the exact global minimum is known: x = [1, 1, 1, 1], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("36. Colville Function"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
        """)

        self.set_grid(-10., 10., sh=True, sh_out=True)

        self.set_min(x=1., y=0.)

    @property
    def args_constr(self):
        return {'d': 4}

    @property
    def ref(self):
        i = [5, 3, 9, 11]
        return np.array(i, dtype=int), 424894.8557820549

    def target(self, x):
        x1, x2, x3, x4 = x

        y1 = 100. * (x1 - x2**2)**2
        y2 = (1 - x1)**2
        y3 = 90. * (x4 - x3**2)**2
        y4 = (1 - x3)**2
        y5 = 10.1 * (x2 - 1)**2
        y6 = 10.1 * (x4 - 1)**2
        y7 = 19.8 * (x2 - 1) * (x4 - 1)

        return y1 + y2 + y3 + y4 + y5 + y6 + y7
