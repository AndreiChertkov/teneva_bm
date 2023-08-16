import numpy as np
from teneva_bm import Bm


class BmFuncAckley(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Ackley function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-32.768, 32.768] (with small shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("1. Ackley 1 Function"; Continuous, Differentiable, Non-separable,
            Scalable, Multimodal).
            See also https://www.sfu.ca/~ssurjano/ackley.html for details (note
            that we use opt_a, opt_b, opt_c and grid limits from this link).
        """)

        self.set_grid(-32.768, +32.768, sh=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def is_func(self):
        return True

    @property
    def opts_info(self):
        return {**super().opts_info,
            'opt_a': {
                'desc': 'Param "a" for Ackley function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 20
            },
            'opt_b': {
                'desc': 'Param "b" for Ackley function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 0.2
            },
            'opt_c': {
                'desc': 'Param "c" for Ackley function',
                'kind': 'float',
                'form': '.6f',
                'dflt': 2.*np.pi
            }
        }

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 21.24996347509561

    def target_batch(self, X):
        y1 = np.sqrt(np.sum(X**2, axis=1) / self.d)
        y1 = -self.opt_a * np.exp(-self.opt_b * y1)

        y2 = np.sum(np.cos(self.opt_c * X), axis=1)
        y2 = -np.exp(y2 / self.d)

        y3 = self.opt_a + np.exp(1.)

        return y1 + y2 + y3

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        par_a = torch.tensor(self.opt_a)
        par_b = torch.tensor(self.opt_b)
        par_c = torch.tensor(self.opt_c)

        y1 = torch.sqrt(torch.sum(x**2) / d)
        y1 = - par_a * torch.exp(-par_b * y1)

        y2 = torch.sum(torch.cos(par_c * x))
        y2 = - torch.exp(y2 / d)

        y3 = par_a + torch.exp(torch.tensor(1.))

        return y1 + y2 + y3
