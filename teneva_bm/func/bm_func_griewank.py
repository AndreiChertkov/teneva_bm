import numpy as np
import teneva
from teneva_bm.func.func import Func


class BmFuncGriewank(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Griewank function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-100, 100] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("59. Griewank Function"; Continuous, Differentiable, Non-Separable,
            Scalable, Multimodal).
            See also https://www.sfu.ca/~ssurjano/griewank.html for details.
        """)

        self.set_grid(-100., +100., sh=True)

        self.set_min(x=0., y=0.)

    @property
    def ref(self):
        return self.ref_i, 9.13373807946276

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Y = self.cores_mul([np.cos(x/np.sqrt(i)) for i,x in enumerate(X.T, 1)])
        Y[-1] *= -1
        return teneva.add(Y, self.cores_add([x**2 / 4000. for x in X.T], a0=1))

    def target_batch(self, X):
        y1 = np.sum(X**2, axis=1) / 4000

        y2 = np.cos(X / np.sqrt(np.arange(1, self.d+1)))
        y2 = - np.prod(y2, axis=1)

        y3 = 1.

        return y1 + y2 + y3

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)

        y1 = torch.sum(x**2) / 4000

        y2 = torch.cos(x / torch.sqrt(torch.arange(d) + 1.))
        y2 = - torch.prod(y2)

        y3 = 1.

        return y1 + y2 + y3
