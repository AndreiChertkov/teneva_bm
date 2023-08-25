import numpy as np
from teneva_bm.func.func import Func


class BmFuncRastrigin(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Rastrigin function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-5.12, 5.12] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Johannes M Dieterich, Bernd Hartke. "Empirical review
            of standard benchmark functions using evolutionary global
            optimization". Applied Mathematics 2012; 3:1552-1564.
            See also https://www.sfu.ca/~ssurjano/rastr.html for details.
        """)

        self.set_grid(-5.12, +5.12, sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'opt_A': {
                'desc': 'Param "A" for Rastrigin function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 10.
            }
        }

    @property
    def opts_plot(self):
        return {'dy_min': 25., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 166.75702361466605

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        return self.cores_add(
            [x**2 - self.opt_A * np.cos(2 * np.pi * x) for x in X.T],
            a0=self.opt_A*self.d)

    def target_batch(self, X):
        y1 = self.opt_A * self.d
        y2 = np.sum(X**2 - self.opt_A * np.cos(2. * np.pi * X), axis=1)
        return y1 + y2

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        opt_A = torch.tensor(self.opt_A)
        pi = torch.tensor(np.pi)

        y1 = opt_A * d
        y2 = torch.sum(x**2 - opt_A * torch.cos(2. * pi * x))

        return y1 + y2
