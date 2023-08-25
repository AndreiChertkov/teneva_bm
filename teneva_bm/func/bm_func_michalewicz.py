import numpy as np
from teneva_bm.func.func import Func


class BmFuncMichalewicz(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Michalewicz function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [0, pi] (with small random shift).
            See Charlie Vanaret, Jean-Baptiste Gotteland, Nicolas Durand,
            Jean-Marc Alliot. "Certified global minima for a benchmark of
            difficult optimization problems". arXiv:2003.09867 2020.
            (the function has d! local minima, and it is multimodal).
            See also https://www.sfu.ca/~ssurjano/michal.html for details.
        """)

        self.set_grid(0., np.pi, sh=True)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'opt_M': {
                'desc': 'Param "M" for Michalewicz function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 10.
            }
        }

    @property
    def opts_plot(self):
        return {'dy_min': 1., 'dy_max': 1.}

    @property
    def ref(self):
        return self.ref_i, -1.2753489806268878

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Y = self.cores_add(
            [np.sin(x) * np.sin(i*x**2/np.pi)**(2*self.opt_M)
                for i, x in enumerate(X.T, 1)])
        Y[-1] *= -1.
        return Y

    def target_batch(self, X):
        y1 = np.sin(np.arange(1, self.d+1) * X**2 / np.pi)
        y = -np.sum(np.sin(X) * y1**(2 * self.opt_M), axis=1)
        return y

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        opt_M = torch.tensor(self.opt_M)
        pi = torch.tensor(np.pi)
        y1 = torch.sin(((torch.arange(d) + 1) * x**2 / pi))
        y = -torch.sum(torch.sin(x) * y1**(2 * opt_M))
        return y
