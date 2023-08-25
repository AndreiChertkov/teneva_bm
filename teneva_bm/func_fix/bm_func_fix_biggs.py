import numpy as np
from teneva_bm.func_fix.func_fix import FuncFix


class BmFuncFixBiggs(FuncFix):
    def __init__(self, d=5, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Biggs function (continuous).
            The dimension is 5 and the mode size may be any (default is n=16),
            Default grid limits are [0, 20] (with small shift);
            the exact global minimum is known: x = [1, 10, 1, 5, 4], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("14. Biggs EXP Function 5"; Continuous, Differentiable,
            Non-separable, Non-scalable, Multimodal).
        """)

        self.set_grid(0., 20., sh=True, sh_out=True)

        self.set_min(x=np.array([1., 10., 1., 5., 4.]), y=0.)

    @property
    def args_constr(self):
        return {'d': 5}

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14]
        return np.array(i, dtype=int), 42.09937733487435

    def target(self, x):
        x1, x2, x3, x4, x5 = x
        i = np.arange(1, 12, dtype=float)
        t = 0.1 * i
        y1 = x3 * np.exp(-t*x1) - x4 * np.exp(-t*x2) + 3 * np.exp(-t*x5)
        y2 = np.exp(-t) - 5. * np.exp(-10.*t) + 3. * np.exp(-4.*t)
        return np.sum((y1-y2)**2)
