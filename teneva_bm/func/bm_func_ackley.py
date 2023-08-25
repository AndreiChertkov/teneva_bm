import numpy as np
from teneva_bm.func.func import Func


class BmFuncAckley(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Ackley function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-32.768, 32.768] (with small shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("1. Ackley Function 1"; Continuous, Differentiable, Non-separable,
            Scalable, Multimodal).
            See also https://www.sfu.ca/~ssurjano/ackley.html for details (note
            that we use opt_A, opt_B, opt_C and grid limits from this link).
        """)

        self.set_grid(-32.768, +32.768, sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'opt_A': {
                'desc': 'Param "A" for Ackley function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 20.
            },
            'opt_B': {
                'desc': 'Param "B" for Ackley function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 0.2
            },
            'opt_C': {
                'desc': 'Param "C" for Ackley function',
                'kind': 'float',
                'form': '.6f',
                'dflt': 2.*np.pi
            }
        }

    @property
    def opts_plot(self):
        return {'dy_min': 2., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 21.24996347509561

    def target_batch(self, X):
        y1 = np.sqrt(np.sum(X**2, axis=1) / self.d)
        y1 = -self.opt_A * np.exp(-self.opt_B * y1)

        y2 = np.sum(np.cos(self.opt_C * X), axis=1)
        y2 = -np.exp(y2 / self.d)

        y3 = self.opt_A + np.exp(1.)

        return y1 + y2 + y3

    def _target_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        opt_A = torch.tensor(self.opt_A)
        opt_B = torch.tensor(self.opt_B)
        opt_C = torch.tensor(self.opt_C)

        y1 = torch.sqrt(torch.sum(x**2) / d)
        y1 = -v * torch.exp(-opt_B * y1)

        y2 = torch.sum(torch.cos(opt_C * x))
        y2 = -torch.exp(y2 / d)

        y3 = opt_A + torch.exp(torch.tensor(1.))

        return y1 + y2 + y3
