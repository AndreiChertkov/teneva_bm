import numpy as np
from teneva_bm.func_fix.func_fix import FuncFix


class BmFuncFixCantrell(FuncFix):
    def __init__(self, d=4, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Cantrell function (continuous).
            The dimension is 4 and the mode size may be any (default is n=16),
            Default grid limits are [-1, 1] (with small shift);
            the exact global minimum value is known: x = [0, 1, 1, 1], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("73. Miele Cantrell Function"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
        """)

        self.set_grid(-1., +1., sh=True, sh_out=True)

        self.set_min(x=np.array([0., 1., 1., 1.]), y=0.)

    @property
    def args_constr(self):
        return {'d': 4}

    @property
    def ref(self):
        i = [5, 3, 9, 11]
        return np.array(i, dtype=int), 220.47873089341215

    def target(self, x):
        x1, x2, x3, x4 = x

        y1 = (np.exp(-x1) - x2)**4
        y2 = 100 * (x2 - x3)**6
        y3 = (np.tan(x3 - x4))**4
        y4 = x1**8

        return y1 + y2 + y3 + y4
