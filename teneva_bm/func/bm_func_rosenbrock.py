import numpy as np
import teneva
from teneva_bm import Bm


class BmFuncRosenbrock(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Rosenbrock function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-2.048, 2.048] (with small random shift);
            the exact global minimum is known: x = [1, ..., 1], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("105. Rosenbrock Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Unimodal).
            See also https://www.sfu.ca/~ssurjano/rosen.html for details (note
            that we use grid limits from this link).
        """)

        self.set_grid(-2.048, +2.048, sh=True, sh_out=True)

        self.set_min(x=[1.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 5130.1430415221985

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Y = []
        for i, x in enumerate(X.T):
            x2 = x*x
            if i == 0:
                G = np.zeros([1, len(x), 3])
                G[0, :, 0] = 1
                G[0, :, 1] = x2
                G[0, :, 2] = 100*(x2**2) + (1-x)**2
            elif i == self.d-1:
                G = np.zeros([3, len(x), 1])
                G[2, :, 0] = 1
                G[1, :, 0] = -200*x
                G[0, :, 0] = 100*x2
            else:
                G = np.zeros([3, len(x), 3])
                G[0, :, 0] = 1.
                G[2, :, 2] = 1.
                G[0, :, 1] = x2
                G[0, :, 2] = 100*x2 + 100*(x2**2) + (1-x)**2
                G[1, :, 2] = -200*x
            Y.append(G)
        return Y

    def target_batch(self, X):
        y1 = 100. * (X[:, 1:] - X[:, :-1]**2)**2
        y2 = (X[:, :-1] - 1.)**2
        return np.sum(y1 + y2, axis=1)

    def _target_pt(self, x):
        """Draft."""
        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2
        return torch.sum(y1 + y2)
