import numpy as np
import teneva
from teneva_bm.func.func import Func


class BmFuncMichalewicz(Func):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Michalewicz function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [0, pi] (with small random shift);
            the exact global minimum is known only for the case of dimensions
            2, 5, and 10; in this cases, only the corresponding value of the
            function is known, but not the argument (except the 2D case).
            See Charlie Vanaret, Jean-Baptiste Gotteland, Nicolas Durand,
            Jean-Marc Alliot. "Certified global minima for a benchmark of
            difficult optimization problems". arXiv:2003.09867 2020
            (the function has d! local minima, and it is multimodal).
            See also https://www.sfu.ca/~ssurjano/michal.html for details.
        """)

        self.set_grid(0., np.pi, sh=True)

        if self.d == 2:
            self.set_min(x=[2.20, 1.57], y=-1.8013)
        if self.d == 5:
            self.set_min(y=-4.687658)
        if self.d == 10:
            self.set_min(y=-9.66015)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'opt_m': {
                'desc': 'Param "m" for Michalewicz function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 10.
            }
        }

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), -1.2753489806268878

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Y = self.cores_add(
            [np.sin(x) * np.sin(i*x**2/np.pi)**(2*self.opt_m)
                for i, x in enumerate(X.T, 1)])
        Y[-1] *= -1.
        return Y

    def target_batch(self, X):
        y1 = np.sin(np.arange(1, self.d+1) * X**2 / np.pi)
        y = -np.sum(np.sin(X) * y1**(2 * self.opt_m), axis=1)
        return y

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        par_m = torch.tensor(self.opt_m)
        pi = torch.tensor(np.pi)
        y1 = torch.sin(((torch.arange(d) + 1) * x**2 / pi))
        y = -torch.sum(torch.sin(x) * y1**(2 * par_m))
        return y
