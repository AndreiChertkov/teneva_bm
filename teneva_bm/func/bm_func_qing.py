import numpy as np
from teneva_bm.func.func import Func


class BmFuncQing(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Qing function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [0, 500] (with small random shift; note
            also that we limit this function to this domain instead of often
            used [-500, 500] to make sure it has a single global minimum);
            the exact global minimum is known: x_i = \sqrt{i}, y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("98. Qing Function"; Continuous, Differentiable, Separable
            Scalable, Multimodal).
        """)

        self.set_grid(0., +500., sh=True, sh_out=True)

        self.set_min(x=np.sqrt(np.arange(1, self.d+1)), y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 1.E+10, 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 106030804604.16588

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        return self.cores_add([(x**2 - i)**2 for i, x in enumerate(X.T, 1)])

    def target_batch(self, X):
        i = np.arange(1, self.d+1)
        return np.sum((X**2 - i)**2, axis=1)

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        i = torch.arange(1, d+1)
        return torch.sum((x**2 - i)**2)
